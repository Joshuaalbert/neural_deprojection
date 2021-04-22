"""
Input Graph:
    nodes: (positions, properties)
    senders/receivers: K-nearest neighbours
    edges: None
    globals: None

Latent Graph:
    nodes: None
    senders/receivers: None
    edges: None
    globals: encoding

Output Graph:
    nodes: properties
    senders/receivers: same as input graph
    edges: None
    globals: None
"""
import tensorflow as tf
import sonnet as snt
from graph_nets.graphs import GraphsTuple
from graph_nets.blocks import NodeBlock, EdgeBlock, GlobalBlock, ReceivedEdgesToNodesAggregator
from neural_deprojection.graph_net_utils import vanilla_training_loop, TrainOneEpoch, AbstractModule, \
    batch_dataset_set_graph_tuples, get_distribution_strategy, build_log_dir, build_checkpoint_dir, histogramdd
from neural_deprojection.models.identify_medium.generate_data import decode_examples
import glob, os, json


def unsorted_segment_scaled_softmax(data,
                                    segment_ids,
                                    num_segments,
                                    name="unsorted_segment_scaled_softmax"):
    """Performs an elementwise softmax operation along segments of a tensor, scaled by 1/sqrt(segment_size).

    The input parameters are analogous to `tf.math.unsorted_segment_sum`. It
    produces an output of the same shape as the input data, after performing an
    elementwise sofmax operation between all of the rows with common segment id.

    Args:
    data: A tensor with at least one dimension.
    segment_ids: A tensor of indices segmenting `data` across the first
      dimension.
    num_segments: A scalar tensor indicating the number of segments. It should
      be at least `max(segment_ids) + 1`.
    name: A name for the operation (optional).

    Returns:
    A tensor with the same shape as `data` after applying the softmax operation.

    """
    with tf.name_scope(name):
        segment_sizes = tf.math.unsorted_segment_sum(tf.ones_like(data), segment_ids, num_segments)
        segment_sizes_sqrt = tf.maximum(tf.math.sqrt(segment_sizes), tf.constant(1., dtype=data.dtype))
        data /= tf.gather(segment_sizes_sqrt, segment_ids)
        segment_maxes = tf.math.unsorted_segment_max(data, segment_ids, num_segments)
        maxes = tf.gather(segment_maxes, segment_ids)
        # Possibly refactor to `tf.stop_gradient(maxes)` for better performance.
        data -= maxes
        exp_data = tf.exp(data)
        segment_sum_exp_data = tf.math.unsorted_segment_sum(exp_data, segment_ids, num_segments)
        sum_exp_data = tf.gather(segment_sum_exp_data, segment_ids)
        return exp_data / sum_exp_data


def received_edges_attention_normaliser(graph,
                                        name="received_edges_attention_normaliser"):
    """Performs elementwise normalization for all received edges by a given node.

    Args:
      graph: A graph containing edge information.
      normalizer: A normalizer function following the signature of
        `modules._unsorted_segment_softmax`.
      name: A name for the operation (optional).

    Returns:
      A tensor with the resulting normalized edges.

    """
    with tf.name_scope(name):
        return unsorted_segment_scaled_softmax(
            data=graph.edges,
            segment_ids=graph.receivers,
            num_segments=tf.reduce_sum(graph.n_node))


class CoreGraph(AbstractModule):
    """
    Each node represents a multidimensional state.
    Each step applies a local rule to each node and updates the nodes.
    Rule:
        Update the outgoing edges from a node with a message.
        Update nodes by
    """

    def __init__(self, message_size, name=None):
        super(CoreGraph, self).__init__(name=name)
        # bottelneck message
        self._generate_message = EdgeBlock(
            edge_model_fn=lambda: snt.nets.MLP([message_size // 2, message_size], activate_final=True, name='edge_fn'),
            use_edges=True,
            use_sender_nodes=True,
            use_receiver_nodes=False,
            use_globals=True,
            name='generate_message'
        )
        self._attention = EdgeBlock(
            edge_model_fn=lambda: snt.nets.MLP([message_size // 2, 1], activate_final=False, name='edge_fn'),
            use_edges=True,
            use_receiver_nodes=True,
            use_sender_nodes=False,
            use_globals=True,
            name='attention_block')
        # bottelneck state
        self._update_nodes = NodeBlock(
            node_model_fn=lambda: snt.nets.MLP([message_size // 2, message_size], activate_final=True),
            use_received_edges=False,
            use_sent_edges=False,
            use_globals=True,
            name='update_nodes')

    def _build(self, graph, **kwargs):
        message_graph = self._generate_message(graph)
        message_graph = message_graph._replace(edges=message_graph.edges + graph.edges)
        attention_graph = self._attention(message_graph)
        attention_graph = attention_graph._replace(edges=attention_graph.edges)
        attention_weights = received_edges_attention_normaliser(attention_graph)
        attended_message_graph = message_graph._replace(edges=message_graph.edges * attention_weights)

        received_edges_aggregator = ReceivedEdgesToNodesAggregator(reducer=tf.math.unsorted_segment_sum)
        aggregated_attended_values = received_edges_aggregator(attended_message_graph)
        updated_graph = self._update_nodes(message_graph._replace(nodes=aggregated_attended_values))
        return updated_graph


class Decoder(AbstractModule):
    """
    Latent Graph:
        nodes: None
        senders/receivers: same as input graph
        edges: None
        globals: encoding

    Output Graph:
        nodes: properties
        senders/receivers: same as input graph
        edges: None
        globals: None
    """

    def __init__(self, message_size, property_size, name=None):
        super(Decoder, self).__init__(name=name)
        self._first_block = snt.Sequential([
            EdgeBlock(
                edge_model_fn=lambda: snt.nets.MLP([message_size, message_size], activate_final=True),
                use_edges=False, use_sender_nodes=False, use_receiver_nodes=False, use_globals=True,
                name='edge_block'),
            NodeBlock(node_model_fn=lambda: snt.nets.MLP([message_size, message_size]),
                      use_received_edges=True, use_sent_edges=True, use_globals=True,
                      received_edges_reducer=tf.math.unsorted_segment_sqrt_n,
                      sent_edges_reducer=tf.math.unsorted_segment_sqrt_n,
                      name='node_block')
        ])

        self._message_passing_graph = CoreGraph(message_size=message_size, name='core')
        self._property_block = NodeBlock(node_model_fn=lambda: snt.nets.MLP([property_size], activate_final=False),
                                         use_received_edges=True, use_sent_edges=True, use_globals=True,
                                         received_edges_reducer=tf.math.unsorted_segment_sqrt_n,
                                         sent_edges_reducer=tf.math.unsorted_segment_sqrt_n,
                                         name='property_block')

    def _build(self, graph: GraphsTuple, positions: GraphsTuple, num_processing_steps):
        # give edges and globals to graph from nodes
        graph = self._first_block(graph)
        outputs = []
        for i in range(num_processing_steps):
            graph = graph._replace(nodes=tf.concat([positions.nodes, graph.nodes], axis=-1))
            graph = self._message_passing_graph(graph)
            outputs.append(self._property_block(graph))
        return outputs


class Encoder(AbstractModule):
    """
    Let us specify a "auto-regressive" encoder which aims for synchronisation.
    1. message passing
    2. global is updated

    Input Graph:
        nodes: (positions, properties)
        senders/receivers: K-nearest neighbours
        edges: None
        globals: None

    Latent Graph:
        nodes: None
        senders/receivers: None
        edges: None
        globals: encoding
    """

    def __init__(self, message_size, latent_size, name=None):
        super(Encoder, self).__init__(name=name)
        self._first_block = snt.Sequential([
            NodeBlock(
                node_model_fn=lambda: snt.nets.MLP([message_size, message_size], activate_final=True),
                use_sent_edges=False,
                use_received_edges=False,
                use_globals=False,
                use_nodes=True,
                name='rn_node_block'),
            EdgeBlock(edge_model_fn=lambda: snt.nets.MLP([message_size, message_size], activate_final=True),
                      use_edges=False,
                      use_globals=False,
                      use_sender_nodes=True,
                      use_receiver_nodes=True,
                      name='rn_edge_block'),
            GlobalBlock(lambda: snt.nets.MLP([latent_size, latent_size], activate_final=True),
                        use_edges=True,
                        use_globals=False,
                        use_nodes=False,
                        edges_reducer=tf.math.unsorted_segment_mean,
                        name='rn_global_block')
        ])
        self._message_passing_graph = CoreGraph(message_size=message_size, name='core')
        self._global_block = GlobalBlock(lambda: snt.nets.MLP([latent_size, latent_size], activate_final=False),
                                         use_edges=False,
                                         use_globals=True,
                                         use_nodes=True,
                                         nodes_reducer=tf.math.unsorted_segment_mean,
                                         name='global_block')

    def _build(self, graph: GraphsTuple, num_processing_steps):
        # give edges and globals to graph from nodes
        graph = self._first_block(graph)
        output = []
        for i in range(num_processing_steps):
            graph = self._message_passing_graph(graph)
            graph = graph._replace(globals=self._global_block(graph).globals + graph.globals)
            output.append(graph)
        return output


class Model(AbstractModule):
    def __init__(self, input_size, message_size, latent_size, num_processing_steps, name=None):
        super(Model, self).__init__(name=name)
        self._encoder = Encoder(message_size, latent_size, name='encoder')
        self._decoder = Decoder(message_size, input_size, name='decoder')
        self._num_processing_steps = num_processing_steps

    def _build(self, batch, **kwargs):
        (graph, positions) = batch

        image_before, _ = histogramdd(positions.nodes[:,0:2],bins=50, weights=graph.nodes[:,-1])
        image_before -= tf.reduce_min(image_before)
        image_before /= tf.reduce_max(image_before)
        tf.summary.image("xy_image_before", image_before[None, :, :, None], step=self.step)

        encoded_graphs = self._encoder(graph, self._num_processing_steps)
        decoded_graphs = self._decoder(encoded_graphs[-1], positions, self._num_processing_steps)

        image_after, _ = histogramdd(positions.nodes[:,0:2], bins=50, weights=decoded_graphs[-1].nodes[:, -1])
        image_after -= tf.reduce_min(image_after)
        image_after /= tf.reduce_max(image_after)

        tf.summary.image("xy_image_after", image_after[None, :, :, None], step=self.step)
        return encoded_graphs, decoded_graphs


MODEL_MAP = dict(model1=Model)


def build_training(model_type, model_parameters, optimizer_parameters, loss_parameters, strategy=None, **kwargs) -> TrainOneEpoch:
    model_cls = MODEL_MAP[model_type]
    model = model_cls(**model_parameters, **kwargs)

    def build_opt(**kwargs):
        opt_type = kwargs.get('opt_type')
        if opt_type == 'adam':
            learning_rate = kwargs.get('learning_rate', 1e-4)
            opt = snt.optimizers.Adam(learning_rate, beta1=1 - 1 / 100, beta2=1 - 1 / 500)
        else:
            raise ValueError('Opt {} invalid'.format(opt_type))
        return opt

    def build_loss(**loss_parameters):
        def loss(model_outputs, batch):
            (encoded_graphs, decoded_graphs) = model_outputs
            (graph, positions) = batch
            # loss =  mean(-sum_k^2 true[k] * log(pred[k]/true[k]))
            return tf.math.sqrt(tf.reduce_mean(tf.math.square(graph.nodes[:, 3:] - decoded_graphs[-1].nodes)))
        return loss

    loss = build_loss(**loss_parameters)
    opt = build_opt(**optimizer_parameters)

    training = TrainOneEpoch(model, loss, opt, strategy=strategy)

    return training


def build_dataset(data_dir, batch_size):
    tfrecords = glob.glob(os.path.join(data_dir, '*.tfrecords'))
    dataset = tf.data.TFRecordDataset(tfrecords).map(
        lambda record_bytes: decode_examples(record_bytes, node_shape=[4]))
    dataset = dataset.map(
        lambda graph_data_dict, img, c: GraphsTuple(globals=None, edges=None, **graph_data_dict))
    dataset = dataset.map(lambda graph: graph._replace(nodes=tf.concat([graph.nodes[:,:3], tf.math.log(graph.nodes[:,3:])],axis=1)))
    dataset = dataset.map(lambda graph: (graph, graph._replace(nodes=graph.nodes[:, :3])))
    # dataset = batch_dataset_set_graph_tuples(all_graphs_same_size=True, dataset=dataset, batch_size=batch_size)
    return dataset


def main(data_dir, config):
    # Make strategy at the start of your main before any other tf code is run.
    strategy = get_distribution_strategy(use_cpus=True, logical_per_physical_factor=1)

    train_dataset = build_dataset(os.path.join(data_dir, 'train'), batch_size=4)
    test_dataset = build_dataset(os.path.join(data_dir, 'test'), batch_size=4)

    # for (graph, positions) in iter(test_dataset):
    #     print(graph)
    #     break

    with strategy.scope():
        train_one_epoch = build_training(num_processing_steps=4, **config)

    log_dir = build_log_dir('test_log_dir', config)
    checkpoint_dir = build_checkpoint_dir('test_checkpointing', config)
    os.makedirs(checkpoint_dir,exist_ok=True)
    with open(os.path.join(checkpoint_dir, 'config.json'), 'w') as f:
        json.dump(config, f)

    vanilla_training_loop(train_one_epoch=train_one_epoch,
                          training_dataset=train_dataset,
                          test_dataset=test_dataset,
                          num_epochs=1000,
                          early_stop_patience=3,
                          checkpoint_dir=checkpoint_dir,
                          log_dir=log_dir,
                          debug=False)


if __name__ == '__main__':
    config = dict(model_type='model1',
                  model_parameters=dict(message_size=8, latent_size=16, input_size=1),
                  optimizer_parameters=dict(learning_rate=1e-5, opt_type='adam'),
                  loss_parameters=dict())
    main('../identify_medium/data', config)

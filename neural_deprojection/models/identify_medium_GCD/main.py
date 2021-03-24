import sys

sys.path.insert(1, '/home/s2675544/git/neural_deprojection/')

from neural_deprojection.graph_net_utils import vanilla_training_loop, TrainOneEpoch, AbstractModule, get_distribution_strategy, build_log_dir, build_checkpoint_dir
import glob, os
import tensorflow as tf
from functools import partial
from graph_nets import blocks
import sonnet as snt
from tensorflow_addons.image import gaussian_filter2d
from graph_nets.graphs import GraphsTuple
from graph_nets.modules import _unsorted_segment_softmax, _received_edges_normalizer
from graph_nets.utils_tf import fully_connect_graph_dynamic
from typing import Callable, Iterable, Optional, Text
from sonnet.src import initializers
from sonnet.src import linear
import matplotlib.pyplot as plt


class RelationNetwork(AbstractModule):
    """Implementation of a Relation Network.

    See https://arxiv.org/abs/1706.01427 for more details.

    The global and edges features of the input graph are not used, and are
    allowed to be `None` (the receivers and senders properties must be present).
    The output graph has updated, non-`None`, globals.
    """

    def \
            __init__(self,
                 edge_model_fn,
                 global_model_fn,
                 reducer=tf.math.unsorted_segment_mean, #try with mean instead of sum
                 use_globals=False,
                 name="relation_network"):
        """Initializes the RelationNetwork module.

        Args:
          edge_model_fn: A callable that will be passed to EdgeBlock to perform
            per-edge computations. The callable must return a Sonnet module (or
            equivalent; see EdgeBlock for details).
          global_model_fn: A callable that will be passed to GlobalBlock to perform
            per-global computations. The callable must return a Sonnet module (or
            equivalent; see GlobalBlock for details).
          reducer: Reducer to be used by GlobalBlock to aggregate edges. Defaults to
            tf.math.unsorted_segment_sum.
          name: The module name.
        """
        super(RelationNetwork, self).__init__(name=name)

        self._edge_block = blocks.EdgeBlock(
            edge_model_fn=edge_model_fn,
            use_edges=False,
            use_receiver_nodes=True,
            use_sender_nodes=True,
            use_globals=use_globals)

        self._global_block = blocks.GlobalBlock(
            global_model_fn=global_model_fn,
            use_edges=True,
            use_nodes=False,
            use_globals=use_globals,
            edges_reducer=reducer)

    def _build(self, graph):
        """Connects the RelationNetwork.

        Args:
          graph: A `graphs.GraphsTuple` containing `Tensor`s, except for the edges
            and global properties which may be `None`.

        Returns:
          A `graphs.GraphsTuple` with updated globals.

        Raises:
          ValueError: If any of `graph.nodes`, `graph.receivers` or `graph.senders`
            is `None`.
        """
        # print(f'Nodes before edge block: {graph.nodes.shape}')
        edge_block = self._edge_block(graph)
        # print(f'Edges after edge block: {edge_block.edges.shape}')
        # print(f'This is the maximum value of the edge block :{np.max(np.array(edge_block.edges))}')
        # print(f'This is the minimum value of the edge block :{np.min(np.array(edge_block.edges))}')
        # print(f'Mean of all edges : {np.mean(np.array(edge_block.edges), axis=0)}')
        output_graph = self._global_block(edge_block)
        # print(f'Globals after globals block: {output_graph.globals.shape}')
        # print(f'This is the global block :{np.array(output_graph.globals)}')
        return output_graph  # graph.replace(globals=output_graph.globals)


class SelfAttention(AbstractModule):
  """Multi-head self-attention module.
  The module is based on the following three papers:
   * A simple neural network module for relational reasoning (RNs):
       https://arxiv.org/abs/1706.01427
   * Non-local Neural Networks: https://arxiv.org/abs/1711.07971.
   * Attention Is All You Need (AIAYN): https://arxiv.org/abs/1706.03762.
  The input to the modules consists of a graph containing values for each node
  and connectivity between them, a tensor containing keys for each node
  and a tensor containing queries for each node.
  The self-attention step consist of updating the node values, with each new
  node value computed in a two step process:
  - Computing the attention weights between each node and all of its senders
   nodes, by calculating sum(sender_key*receiver_query) and using the softmax
   operation on all attention weights for each node.
  - For each receiver node, compute the new node value as the weighted average
   of the values of the sender nodes, according to the attention weights.
  - Nodes with no received edges, get an updated value of 0.
  Values, keys and queries contain a "head" axis to compute independent
  self-attention for each of the heads.
  """

  def __init__(self, name="self_attention"):
    """Inits the module.
    Args:
      name: The module name.
    """
    super(SelfAttention, self).__init__(name=name)
    self._normalizer = _unsorted_segment_softmax

  def _build(self, node_values, node_keys, node_queries, attention_graph):
    """Connects the multi-head self-attention module.
    The self-attention is only computed according to the connectivity of the
    input graphs, with receiver nodes attending to sender nodes.
    Args:
      node_values: Tensor containing the values associated to each of the nodes.
        The expected shape is [total_num_nodes, num_heads, key_size].
      node_keys: Tensor containing the key associated to each of the nodes. The
        expected shape is [total_num_nodes, num_heads, key_size].
      node_queries: Tensor containing the query associated to each of the nodes.
        The expected shape is [total_num_nodes, num_heads, query_size]. The
        query size must be equal to the key size.
      attention_graph: Graph containing connectivity information between nodes
        via the senders and receivers fields. Node A will only attempt to attend
        to Node B if `attention_graph` contains an edge sent by Node A and
        received by Node B.
    Returns:
      An output `graphs.GraphsTuple` with updated nodes containing the
      aggregated attended value for each of the nodes with shape
      [total_num_nodes, num_heads, value_size].
    Raises:
      ValueError: if the input graph does not have edges.
    """

    # Sender nodes put their keys and values in the edges.
    # [total_num_edges, num_heads, query_size]
    sender_keys = blocks.broadcast_sender_nodes_to_edges(
        attention_graph.replace(nodes=node_keys))
    # [total_num_edges, num_heads, value_size]
    sender_values = blocks.broadcast_sender_nodes_to_edges(
        attention_graph.replace(nodes=node_values))

    # Receiver nodes put their queries in the edges.
    # [total_num_edges, num_heads, key_size]
    receiver_queries = blocks.broadcast_receiver_nodes_to_edges(
        attention_graph.replace(nodes=node_queries))

    # Attention weight for each edge.
    # [total_num_edges, num_heads]
    attention_weights_logits = tf.reduce_sum(
        sender_keys * receiver_queries, axis=-1)
    normalized_attention_weights = _received_edges_normalizer(
        attention_graph.replace(edges=attention_weights_logits),
        normalizer=self._normalizer)

    # Attending to sender values according to the weights.
    # [total_num_edges, num_heads, embedding_size]
    attented_edges = sender_values * normalized_attention_weights[..., None]

    # Summing all of the attended values from each node.
    # [total_num_nodes, num_heads, embedding_size]
    received_edges_aggregator = blocks.ReceivedEdgesToNodesAggregator(
        reducer=tf.math.unsorted_segment_sum)
    aggregated_attended_values = received_edges_aggregator(
        attention_graph.replace(edges=attented_edges))

    return attention_graph.replace(nodes=aggregated_attended_values)

class Autoencoder(AbstractModule):
    def __init__(self, name=None):
        super(Autoencoder, self).__init__(name=name)
        self.encoder = snt.Sequential([snt.Conv2D(2, kernel_size, stride=2, padding='SAME'), tf.nn.relu,
                                      snt.Conv2D(4, kernel_size, stride=2, padding='SAME'), tf.nn.relu,
                                      snt.Conv2D(8, kernel_size, stride=2, padding='SAME'), tf.nn.relu,
                                      snt.Conv2D(16, kernel_size, stride=2, padding='SAME'), tf.nn.relu,
                                      snt.Conv2D(32, kernel_size, stride=2, padding='SAME'), tf.nn.relu,
                                       snt.Conv2D(64, kernel_size, stride=2, padding='SAME'), tf.nn.relu])

        self.decoder = snt.Sequential([snt.Conv2DTranspose(64, kernel_size, stride=2, padding='SAME'), tf.nn.relu,
                                       snt.Conv2DTranspose(32, kernel_size, stride=2, padding='SAME'), tf.nn.relu,
                                       snt.Conv2DTranspose(16, kernel_size, stride=2, padding='SAME'), tf.nn.relu,
                                       snt.Conv2DTranspose(8, kernel_size, stride=2, padding='SAME'), tf.nn.relu,
                                       snt.Conv2DTranspose(4, kernel_size, stride=2, padding='SAME'), tf.nn.relu,
                                       snt.Conv2DTranspose(2, kernel_size, stride=2, padding='SAME'), tf.nn.relu,
                                       snt.Conv2D(1, kernel_size, padding='SAME')])

    @property
    def step(self):
        if self._step is None:
            raise ValueError("Need to set step idx variable. model.step = epoch")
        return self._step

    @step.setter
    def step(self, value):
        self._step = value

    def _build(self, batch):
        (img, ) = batch
        img = gaussian_filter2d(img, filter_shape=[6, 6])
        img_before_autoencoder = (img - tf.reduce_min(img)) / (
                tf.reduce_max(img) - tf.reduce_min(img))
        tf.summary.image(f'img_before_autoencoder', img_before_autoencoder, step=self.step)
        encoded_img = self.encoder(img)

        print(encoded_img.shape)

        decoded_img = self.decoder(encoded_img)
        img_after_autoencoder = (decoded_img - tf.reduce_min(decoded_img)) / (
                tf.reduce_max(decoded_img) - tf.reduce_min(decoded_img))
        tf.summary.image(f'img_after_autoencoder', img_after_autoencoder, step=self.step)
        return decoded_img


class Model(AbstractModule):
    """Model inherits from AbstractModule, which contains a __call__ function which executes a _build function
    that is to be specified in the child class. So for example:
    model = Model(), then model() returns the output of _build()

    AbstractModule inherits from snt.Module, which has useful functions that can return the (trainable) variables,
    so the Model class has this functionality as well
    An instance of the RelationNetwork class also inherits from AbstractModule,
    so it also executes its _build() function when called and it can return its (trainable) variables


    A RelationNetwork contains an edge block and a global block:

    The edge block generally uses the edge, receiver, sender and global attributes of the input graph
    to calculate the new edges.
    In our case we currently only use the receiver and sender attributes to calculate the edges.

    The global block generally uses the aggregated edge, aggregated node and the global attributes of the input graph
    to calculate the new globals.
    In our case we currently only use the aggregated edge attributes to calculate the new globals.


    As input the RelationNetwork needs two (neural network) functions:
    one to calculate the new edges from receiver and sender nodes
    and one to calculate the globals from the aggregated edges.

    The new edges will be a vector with size 16 (i.e. the output of the first function in the RelationNetwork)
    The new globals will also be a vector with size 16 (i.e. the output of the second function in the RelationNetwork)

    The image_cnn downscales the image (currently from 4880x4880 to 35x35) and encodes the image in 16 channels.
    So we (currently) go from (4880,4880,1) to (35,35,16)
    """

    def __init__(self, mlp_layers=2, mlp_layer_nodes=32, conv_layers=6, kernel_size=4, image_feature_size=16, name=None):
        super(Model, self).__init__(name=name)
        self.attention_graph = SelfAttention()
        self.attention_image = SelfAttention()
        self.encoder_graph = RelationNetwork(lambda: snt.nets.MLP(mlp_layers * [mlp_layer_nodes] + [16], activate_final=True),
                                       lambda: snt.nets.MLP(mlp_layers * [mlp_layer_nodes] + [16], activate_final=True))
        self.encoder_image = RelationNetwork(lambda: snt.nets.MLP(mlp_layers * [mlp_layer_nodes] + [16], activate_final=True),
                                       lambda: snt.nets.MLP(mlp_layers * [mlp_layer_nodes] + [16], activate_final=True))
        seq_argument = []
        for _ in range(conv_layers - 1):
            seq_argument.append(snt.Conv2D(16, kernel_size, stride=2, padding='valid'))
            seq_argument.append(tf.nn.relu)

        self.image_cnn = snt.Sequential(seq_argument +
                                        [snt.Conv2D(image_feature_size, kernel_size, stride=2, padding='valid'), tf.nn.relu])
        self.compare = snt.nets.MLP([32, 1])
        self.image_feature_size = image_feature_size

    @property
    def step(self):
        if self._step is None:
            raise ValueError("Need to set step idx variable. model.step = epoch")
        return self._step

    @step.setter
    def step(self, value):
        self._step = value

    def _build(self, batch, *args, **kwargs):
        (graph, img, c) = batch
        del c

        print(f'Node example : {graph.nodes[0]}')
        # print(f'Virtual particles in cluster : {graph.nodes.shape}')
        # print(f'Original image shape : {img.shape}')
        # print(f'Image : {tf.reduce_mean(img)}')
        # print(f'St dev : {(tf.reduce_mean(img**2) - tf.reduce_mean(img)**2)**(1/2)}')
        # print(f'Min : {tf.reduce_min(img)}')
        # print(f'Max : {tf.reduce_max(img)}')

        attended_graph = self.attention_graph(graph.nodes,
                                              graph.nodes,
                                              graph.nodes,
                                              graph)
        encoded_graph = self.encoder_graph(attended_graph)

        print("IMG SHAPE:", img[None, ...].shape)
        print("IMG MIN MAX:", tf.math.reduce_min(img[None, ...]), tf.math.reduce_max(img[None, ...]))

        img_before_cnn = img[None, ...]
        img_before_cnn = (img_before_cnn - tf.reduce_min(img_before_cnn)) / (tf.reduce_max(img_before_cnn) - tf.reduce_min(img_before_cnn))
        tf.summary.image(f'img_before_cnn', img_before_cnn, step=self.step)

        img = self.image_cnn(img[None, ...])  # 1, w,h,c -> w*h, c

        print("IMG SHAPE AFTER CNN:", tf.transpose(img, [3, 1, 2, 0]).shape)
        print("IMG MIN MAX AFTER CNN:", tf.math.reduce_min(tf.transpose(img, [3, 1, 2, 0])), tf.math.reduce_max(tf.transpose(img, [3, 1, 2, 0])))

        tf.summary.image(f'img_after_cnn', tf.transpose(img, [3, 1, 2, 0]), step=self.step)

        img_nodes = tf.reshape(img, (-1, self.image_feature_size))
        img_graph = GraphsTuple(nodes=img_nodes,
                            edges=None,
                            globals=None,
                            receivers=None,
                            senders=None,
                            n_node=tf.shape(img_nodes)[0:1],
                            n_edge=tf.constant([0]))
        connected_graph = fully_connect_graph_dynamic(img_graph)

        attended_img = self.attention_graph(connected_graph.nodes,
                                            connected_graph.nodes,
                                            connected_graph.nodes,
                                            connected_graph)
        encoded_img = self.encoder_image(attended_img)

        # print(f'Convolutional network output shape : {img.shape}')
        # print(f'Encoded particle graph nodes shape : {encoded_graph.nodes.shape}')
        # print(f'Encoded particle graph edges shape : {encoded_graph.edges.shape}')
        # print(f'Encoded image graph nodes shape : {encoded_img.nodes.shape}')
        # print(f'Encoded image graph edges shape : {encoded_img.edges.shape}')
        # print(f'Encoded particle graph globals : {encoded_graph.globals}')
        # print(f'Encoded image graph globals : {encoded_img.globals}')

        distance = self.compare(tf.concat([encoded_graph.globals, encoded_img.globals], axis=1)) + self.compare(tf.concat([encoded_img.globals, encoded_graph.globals], axis=1))
        return distance

MODEL_MAP = dict(model1=Model)


def build_training(model_type, model_parameters, optimizer_parameters, loss_parameters, strategy=None) -> TrainOneEpoch:
    model_cls = MODEL_MAP[model_type]

    model = model_cls(**model_parameters)

    def build_opt(**kwargs):
        opt_type = kwargs.get('opt_type')
        if opt_type == 'adam':
            learning_rate = kwargs.get('learning_rate', 1e-4)
            opt = snt.optimizers.Adam(learning_rate)
        else:
            raise ValueError('Opt {} invalid'.format(opt_type))
        return opt


    def build_loss(**loss_parameters):
        def loss(model_outputs, batch):
            (graph, img, c) = batch
            # loss =  mean(-sum_k^2 true[k] * log(pred[k]/true[k]))
            return tf.reduce_mean(tf.losses.binary_crossentropy(c[None, None], model_outputs, from_logits=True))# tf.math.sqrt(tf.reduce_mean(tf.math.square(rank - tf.nn.sigmoid(model_outputs[:, 0]))))
        return loss

    loss = build_loss(**loss_parameters)
    opt = build_opt(**optimizer_parameters)

    training = TrainOneEpoch(model, loss, opt, strategy=strategy)

    return training


def feature_to_graph_tuple(name=''):
    return {f'{name}_nodes': tf.io.FixedLenFeature([], dtype=tf.string),
            f'{name}_edges': tf.io.FixedLenFeature([], dtype=tf.string),
            f'{name}_senders': tf.io.FixedLenFeature([], dtype=tf.string),
            f'{name}_receivers': tf.io.FixedLenFeature([], dtype=tf.string)}


def decode_examples(record_bytes, node_shape=None, edge_shape=None, image_shape=None):
    """
    Decodes raw bytes as returned from tf.data.TFRecordDataset([example_path]) into a GraphTuple and image
    Args:
        record_bytes: raw bytes
        node_shape: shape of nodes if known.
        edge_shape: shape of edges if known.
        image_shape: shape of image if known.

    Returns: (GraphTuple, image)
    """
    parsed_example = tf.io.parse_single_example(
        # Data
        record_bytes,

        # Schema
        dict(
            image=tf.io.FixedLenFeature([], dtype=tf.string),
            cluster_idx=tf.io.FixedLenFeature([], dtype=tf.string),
            projection_idx=tf.io.FixedLenFeature([], dtype=tf.string),
            vprime=tf.io.FixedLenFeature([], dtype=tf.string),
            **feature_to_graph_tuple('graph')
        )
    )
    image = tf.io.parse_tensor(parsed_example['image'], tf.float32)

    # image = tf.math.log(image / 43.) + tf.math.log(0.5)
    image.set_shape(image_shape)
    vprime = tf.io.parse_tensor(parsed_example['vprime'], tf.float32)
    vprime.set_shape((3, 3))
    cluster_idx = tf.io.parse_tensor(parsed_example['cluster_idx'], tf.int32)
    cluster_idx.set_shape(())
    projection_idx = tf.io.parse_tensor(parsed_example['projection_idx'], tf.int32)
    projection_idx.set_shape(())
    graph_nodes = tf.io.parse_tensor(parsed_example['graph_nodes'], tf.float32)
    if node_shape is not None:
        graph_nodes.set_shape([None] + list(node_shape))
    graph_edges = tf.io.parse_tensor(parsed_example['graph_edges'], tf.float32)
    if edge_shape is not None:
        graph_edges.set_shape([None] + list(edge_shape))
    receivers = tf.io.parse_tensor(parsed_example['graph_receivers'], tf.int64)
    receivers.set_shape([None])
    senders = tf.io.parse_tensor(parsed_example['graph_senders'], tf.int64)
    senders.set_shape([None])
    graph = GraphsTuple(nodes=graph_nodes,
                        edges=graph_edges,
                        globals=tf.zeros([1]),
                        receivers=receivers,
                        senders=senders,
                        n_node=tf.shape(graph_nodes)[0:1],
                        n_edge=tf.shape(graph_edges)[0:1])
    return (graph, image, cluster_idx, projection_idx, vprime)


def build_dataset(tfrecords):
    # Extract the dataset (graph tuple, image, example_idx) from the tfrecords files
    dataset = tf.data.TFRecordDataset(tfrecords).map(partial(decode_examples,
                                                             node_shape=(10,),
                                                             edge_shape=(2,),
                                                             image_shape=(1000, 1000, 1)))  # (graph, image, idx)
    # Take the graphs and their corresponding index and shuffle the order of these pairs
    # Do the same for the images
    _graphs = dataset.map(
        lambda graph, img, cluster_idx, projection_idx, vprime: (graph, 26 * cluster_idx + projection_idx)).shuffle(
        buffer_size=260)
    _images = dataset.map(
        lambda graph, img, cluster_idx, projection_idx, vprime: (img, 26 * cluster_idx + projection_idx)).shuffle(
        buffer_size=260)

    # Zip the shuffled datasets back together so typically the index of the graph and image don't match.
    shuffled_dataset = tf.data.Dataset.zip((_graphs, _images))  # ((graph, idx1), (img, idx2))

    # Reshape the dataset to the graph, the image and a yes or no whether the indices are the same
    # So ((graph, idx1), (img, idx2)) --> (graph, img, True/False)
    shuffled_dataset = shuffled_dataset.map(lambda ds1, ds2: (ds1[0], ds2[0], ds1[1] == ds2[1]))  # (graph, img, yes/no)

    # Take the subset of the data where the graph and image don't correspond (which is most of the dataset, since it's shuffled)
    shuffled_dataset = shuffled_dataset.filter(lambda graph, img, c: ~c)

    # Transform the True/False class into 1/0 integer
    shuffled_dataset = shuffled_dataset.map(lambda graph, img, c: (graph, img, tf.cast(c, tf.int32)))

    # Use the original dataset where all indices correspond and give them class True and turn that into an integer
    # So every instance gets class 1
    nonshuffeled_dataset = dataset.map(lambda graph, img, cluster_idx, projection_idx, vprime: (
    graph, img, tf.constant(1, dtype=tf.int32)))  # (graph, img, yes)

    # For the training data, take a sample either from the correct or incorrect combinations of graphs and images
    nn_dataset = tf.data.experimental.sample_from_datasets([shuffled_dataset, nonshuffeled_dataset])
    return nn_dataset


def train_identify_medium(data_dir, config):
    strategy = get_distribution_strategy(use_cpus=False, logical_per_physical_factor=11, memory_limit=900)

    # lists containing tfrecord files
    train_tfrecords = glob.glob(os.path.join(data_dir, 'train', '*.tfrecords'))
    test_tfrecords = glob.glob(os.path.join(data_dir, 'test', '*.tfrecords'))

    print(f'Number of training tfrecord files : {len(train_tfrecords)}')
    print(f'Number of test tfrecord files : {len(test_tfrecords)}')
    print(f'Total : {len(train_tfrecords) + len(test_tfrecords)}')

    train_dataset = build_dataset(train_tfrecords)
    test_dataset = build_dataset(test_tfrecords)

    # for (graph, img, c) in iter(test_dataset):
    #     print(graph)
    #     plt.imshow(img)
    #
    #     plt.colorbar()
    #     plt.show()
    #     break

    with strategy.scope():
        train_one_epoch = build_training(**config)

    log_dir = build_log_dir('test_log_dir', config)
    checkpoint_dir = build_checkpoint_dir('test_checkpointing', config)

    vanilla_training_loop(train_one_epoch=train_one_epoch,
                          training_dataset=train_dataset,
                          test_dataset=test_dataset,
                          num_epochs=20,
                          early_stop_patience=3,
                          checkpoint_dir=checkpoint_dir,
                          log_dir=log_dir,
                          debug=False)

def train_autoencoder(data_dir):
    # strategy = get_distribution_strategy(use_cpus=False, logical_per_physical_factor=1, memory_limit=10000)

    # lists containing tfrecord files
    train_tfrecords = glob.glob(os.path.join(data_dir, 'train', '*.tfrecords'))
    test_tfrecords = glob.glob(os.path.join(data_dir, 'test', '*.tfrecords'))

    print(f'Number of training tfrecord files : {len(train_tfrecords)}')
    print(f'Number of test tfrecord files : {len(test_tfrecords)}')
    print(f'Total : {len(train_tfrecords) + len(test_tfrecords)}')

    train_dataset = build_dataset(train_tfrecords)
    test_dataset = build_dataset(test_tfrecords)

    train_dataset = train_dataset.map(lambda graph, img, c: (img,)).batch(batch_size=32)
    test_dataset = test_dataset.map(lambda graph, img, c: (img,)).batch(batch_size=32)

    # with strategy.scope():
    model = Autoencoder()

    learning_rate = 1e-3
    opt = snt.optimizers.Adam(learning_rate)

    def loss(model_outputs, batch):
        (img,) = batch
        decoded_img = model_outputs
        return tf.reduce_mean((gaussian_filter2d(img, filter_shape=[6, 6]) - decoded_img[:, 12:-12, 12:-12, :]) ** 2)

    train_one_epoch = TrainOneEpoch(model, loss, opt, strategy=None)

    log_dir = 'autoencoder_log_dir'
    checkpoint_dir = 'autoencoder_checkpointing'

    vanilla_training_loop(train_one_epoch=train_one_epoch,
                          training_dataset=train_dataset,
                          test_dataset=test_dataset,
                          num_epochs=40,
                          early_stop_patience=3,
                          checkpoint_dir=checkpoint_dir,
                          log_dir=log_dir,
                          debug=False)

def main(data_dir, config):
    train_autoencoder(data_dir)
    # train_identify_medium(data_dir, config)


if __name__ == '__main__':
    tfrec_base_dir = '/home/s2675544/data/tf_records'
    tfrec_dir = os.path.join(tfrec_base_dir, 'snap_128_tf_records')

    import numpy as np

    learning_rate = 1e-5
    kernel_size = 4
    mlp_layers = 2
    image_feature_size = 32
    conv_layers = 6
    mlp_layer_nodes = 64

    config = dict(model_type='model1',
                  model_parameters=dict(mlp_layers=mlp_layers,
                                        mlp_layer_nodes=mlp_layer_nodes,
                                        conv_layers=conv_layers,
                                        kernel_size=kernel_size,
                                        image_feature_size=image_feature_size),
                  optimizer_parameters=dict(learning_rate=learning_rate, opt_type='adam'),
                  loss_parameters=dict())
    main(tfrec_dir, config)

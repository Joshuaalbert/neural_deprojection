import sys

sys.path.insert(1, '/home/s2675544/git/neural_deprojection/')

from neural_deprojection.graph_net_utils import vanilla_training_loop, TrainOneEpoch, AbstractModule, \
    get_distribution_strategy, build_log_dir, build_checkpoint_dir, batch_dataset_set_graph_tuples
# from neural_deprojection.models.identify_medium.generate_data import graph_tuple_to_feature

import glob, os
import tensorflow as tf
from functools import partial
from graph_nets import blocks
import sonnet as snt
from tensorflow_addons.image import gaussian_filter2d
from graph_nets.graphs import GraphsTuple
from graph_nets.modules import _unsorted_segment_softmax, _received_edges_normalizer, GraphIndependent, SelfAttention, GraphNetwork
from graph_nets.utils_tf import fully_connect_graph_dynamic, concat
from sonnet.src import utils, once
import time
# from typing import Callable, Iterable, Optional, Text
# from sonnet.src import initializers
# from sonnet.src import linear
# import matplotlib.pyplot as plt
# import networkx as nx
import numpy as np


class MultiHeadLinear(AbstractModule):
    """Linear module, optionally including bias."""

    def __init__(self,
                 output_size: int,
                 num_heads: int = 1,
                 with_bias: bool = True,
                 w_init=None,
                 b_init=None,
                 name=None):
        """Constructs a `Linear` module.

        Args:
          output_size: Output dimensionality.
          with_bias: Whether to include bias parameters. Default `True`.
          w_init: Optional initializer for the weights. By default the weights are
            initialized truncated random normal values with a standard deviation of
            `1 / sqrt(input_feature_size)`, which is commonly used when the inputs
            are zero centered (see https://arxiv.org/abs/1502.03167v3).
          b_init: Optional initializer for the bias. By default the bias is
            initialized to zero.
          name: Name of the module.
        """
        super(MultiHeadLinear, self).__init__(name=name)
        self.output_size = output_size
        self.with_bias = with_bias
        self.w_init = w_init
        self.num_heads = num_heads
        if with_bias:
            self.b_init = b_init if b_init is not None else snt.initializers.Zeros()
        elif b_init is not None:
            raise ValueError("When not using a bias the b_init must be None.")

    @once.once
    def _initialize(self, inputs: tf.Tensor):
        """Constructs parameters used by this module."""
        utils.assert_minimum_rank(inputs, 2)

        input_size = inputs.shape[-1]
        if input_size is None:  # Can happen inside an @tf.function.
            raise ValueError("Input size must be specified at module build time.")

        self.input_size = input_size

        if self.w_init is None:
            # See https://arxiv.org/abs/1502.03167v3.
            stddev = 1 / tf.math.sqrt(self.input_size * 1.0)
            self.w_init = snt.initializers.TruncatedNormal(stddev=stddev)

        self.w = tf.Variable(
            self.w_init([self.num_heads, self.input_size, self.output_size], inputs.dtype),
            name="w")

        if self.with_bias:
            self.b = tf.Variable(
                self.b_init([self.num_heads, self.output_size], inputs.dtype), name="b")

    def _build(self, inputs: tf.Tensor) -> tf.Tensor:
        self._initialize(inputs)

        # [num_nodes, node_size].[num_heads, node_size, output_size] -> [num_nodes, num_heads, output_size]
        outputs = tf.einsum('ns,hso->nho', inputs, self.w, optimize='optimal')
        # outputs = tf.matmul(inputs, self.w)
        if self.with_bias:
            outputs = tf.add(outputs, self.b)
        return outputs


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
                     reducer=tf.math.unsorted_segment_mean,
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

        edge_block = self._edge_block(graph)
        output_graph = self._global_block(edge_block)
        return output_graph


class EncodeProcessDecode(AbstractModule):
    """Full encode-process-decode model.
    The model we explore includes three components:
    - An "Encoder" graph net, which independently encodes the edge, node, and
      global attributes (does not compute relations etc.).
    - A "Core" graph net, which performs N rounds of processing (message-passing)
      steps. The input to the Core is the concatenation of the Encoder's output
      and the previous output of the Core (labeled "Hidden(t)" below, where "t" is
      the processing step).
    - A "Decoder" graph net, which independently decodes the edge, node, and
      global attributes (does not compute relations etc.), on each message-passing
      step.
                        Hidden(t)   Hidden(t+1)
                           |            ^
              *---------*  |  *------*  |  *---------*
              |         |  |  |      |  |  |         |
    Input --->| Encoder |  *->| Core |--*->| Decoder |---> Output(t)
              |         |---->|      |     |         |
              *---------*     *------*     *---------*
    """

    def __init__(self,
                 encoder,
                 core,
                 decoder,
                 name="EncodeProcessDecode"):
        super(EncodeProcessDecode, self).__init__(name=name)
        self._encoder = encoder
        self._core = core
        self._decoder = decoder

    def _build(self, input_graph, num_processing_steps, positions):
        latent_graph = self._encoder(input_graph, positions)
        # for _ in range(num_processing_steps):
        #     latent_graph = self._core(latent_graph)

        # state = (counter, latent_graph)
        _, latent_graph = tf.while_loop(cond=lambda const, state: const < num_processing_steps,
                      body=lambda const, state: (const+1, self._core(state, positions)),
                      loop_vars=(tf.constant(0), latent_graph))

        return self._decoder(latent_graph, positions)


class CoreNetwork(AbstractModule):
    """
    Core network which can be used in the EncodeProcessDecode network. Closely follows the transformer architecture.
    """

    def __init__(self,
                 num_heads,
                 multi_head_output_size,
                 input_node_size,
                 name=None):
        super(CoreNetwork, self).__init__(name=name)
        self.num_heads = num_heads
        self.multi_head_output_size = multi_head_output_size

        self.output_linear = snt.Linear(output_size=input_node_size)
        self.FFN = snt.nets.MLP([32, input_node_size], activate_final=False)  # Feed forward network
        self.normalization = lambda x: (x - tf.reduce_mean(x)) / tf.math.reduce_std(x)
        self.ln1 = snt.LayerNorm(axis=1, eps=1e-6, create_scale=True, create_offset=True)
        self.ln2 = snt.LayerNorm(axis=1, eps=1e-6, create_scale=True, create_offset=True)

        self.v_linear = MultiHeadLinear(output_size=multi_head_output_size, num_heads=num_heads)  # values
        self.k_linear = MultiHeadLinear(output_size=multi_head_output_size, num_heads=num_heads)  # keys
        self.q_linear = MultiHeadLinear(output_size=multi_head_output_size, num_heads=num_heads)  # queries
        self.self_attention = SelfAttention()

    def _build(self, latent, positions=None):
        node_values = self.v_linear(latent.nodes)
        node_keys = self.k_linear(latent.nodes)
        node_queries = self.q_linear(latent.nodes)
        attended_latent = self.self_attention(node_values=node_values,
                                              node_keys=node_keys,
                                              node_queries=node_queries,
                                              attention_graph=latent)
        output_nodes = tf.reshape(attended_latent.nodes, (-1, self.num_heads * self.multi_head_output_size))
        output_nodes = self.ln1(self.output_linear(output_nodes) + latent.nodes)
        output_nodes = self.ln2(self.FFN(output_nodes))
        output_graph = latent.replace(nodes=output_nodes)
        if positions is not None:
            prepend_nodes = tf.concat([positions, latent.nodes[:, 3:]], axis=1)
            latent.replace(nodes=prepend_nodes)
        return output_graph


class EncoderNetwork(AbstractModule):
    """
    Encoder network that updates the graph to viable input for the Core network.
    Contains a node block to update the edges and a relation network to generate edges and globals.
    """

    def __init__(self,
                 edge_model_fn,
                 node_model_fn,
                 global_model_fn,
                 name=None):
        super(EncoderNetwork, self).__init__(name=name)
        self.node_block = blocks.NodeBlock(node_model_fn,
                                           use_received_edges=False,
                                           use_sent_edges=False,
                                           use_nodes=True,
                                           use_globals=False)
        self.relation_network = RelationNetwork(edge_model_fn=edge_model_fn,
                                                global_model_fn=global_model_fn)

    def _build(self, input_graph, positions):
        latent = self.node_block(input_graph)

        if positions is not None:
            prepend_nodes = tf.concat([positions, latent.nodes[:, 3:]], axis=1)
            latent.replace(nodes=prepend_nodes)

        output = self.relation_network(latent)
        return output


class DecoderNetwork(AbstractModule):
    """
    Decoder network that updates the nodes of a graph.
    By default only uses the globals to update the nodes, but can also use nodes and edges.
    """

    def __init__(self,
                 node_model_fn,
                 use_received_edges=False,
                 use_sent_edges=False,
                 use_nodes=False,
                 use_globals=True,
                 name=None):
        super(DecoderNetwork, self).__init__(name=name)
        self.node_block = blocks.NodeBlock(node_model_fn,
                                           use_received_edges=use_received_edges,
                                           use_sent_edges=use_sent_edges,
                                           use_nodes=use_nodes,
                                           use_globals=use_globals)

    def _build(self, input_graph, positions):
        output = self.node_block(input_graph)

        if positions is not None:
            prepend_nodes = tf.concat([positions, output.nodes[:, 3:]], axis=1)
            output.replace(nodes=prepend_nodes)

        return output


class GraphAutoEncoder(AbstractModule):
    def __init__(self,
                 mlp_size=16,
                 cluster_encoded_size=10,
                 image_encoded_size=64,
                 num_heads=10,
                 kernel_size=4,
                 image_feature_size=16,
                 core_steps=10, name=None):
        super(GraphAutoEncoder, self).__init__(name=name)

        self.epd_encoder = EncodeProcessDecode(encoder=EncoderNetwork(edge_model_fn=lambda: snt.nets.MLP([mlp_size], activate_final=True),
                                                                      node_model_fn=lambda: snt.Linear(cluster_encoded_size),
                                                                      global_model_fn=lambda: snt.nets.MLP([mlp_size], activate_final=True)),
                                               core=CoreNetwork(num_heads=num_heads,
                                                                multi_head_output_size=cluster_encoded_size,
                                                                input_node_size=cluster_encoded_size),
                                               decoder=EncoderNetwork(edge_model_fn=lambda: snt.nets.MLP([mlp_size], activate_final=True),
                                                                      node_model_fn=lambda: snt.Linear(cluster_encoded_size),
                                                                      global_model_fn=lambda: snt.nets.MLP([32, 32, 64], activate_final=True)))

        self.epd_decoder = EncodeProcessDecode(encoder=DecoderNetwork(node_model_fn=lambda: snt.Linear(cluster_encoded_size)),
                                               core=CoreNetwork(num_heads=num_heads,
                                                                multi_head_output_size=cluster_encoded_size,
                                                                input_node_size=cluster_encoded_size),
                                               decoder=DecoderNetwork(node_model_fn=lambda: snt.Linear(cluster_encoded_size),
                                                                      use_nodes=True))

        self.image_feature_size = image_feature_size

        self._core_steps = core_steps

    @property
    def step(self):
        if self._step is None:
            raise ValueError("Need to set step idx variable. model.step = epoch")
        return self._step

    @step.setter
    def step(self, value):
        self._step = value

    def _build(self, batch, *args, **kwargs):
        graph = batch

        positions = graph.nodes[:, 0:3]

        encoded_graph = self.epd_encoder(graph, self._core_steps, positions)
        decoded_graph = self.epd_decoder(encoded_graph, self._core_steps, positions)

        return decoded_graph


def feature_to_graph_tuple(name=''):
    return {f'{name}_nodes': tf.io.FixedLenFeature([], dtype=tf.string),
            f'{name}_edges': tf.io.FixedLenFeature([], dtype=tf.string),
            f'{name}_senders': tf.io.FixedLenFeature([], dtype=tf.string),
            f'{name}_receivers': tf.io.FixedLenFeature([], dtype=tf.string)}


def decode_examples(record_bytes,
                    node_shape=None,
                    edge_shape=None,
                    image_shape=None):
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
    receivers = tf.cast(receivers, tf.int32)
    senders = tf.io.parse_tensor(parsed_example['graph_senders'], tf.int64)
    senders.set_shape([None])
    senders = tf.cast(senders, tf.int32)
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
        lambda graph, img, cluster_idx, projection_idx, vprime:
        (graph, 26 * cluster_idx + projection_idx)).shuffle(
        buffer_size=260)  # .replace(globals=tf.zeros((1, 1)))
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


def train_ae_3d(data_dir, config):
    train_tfrecords = glob.glob(os.path.join(data_dir, 'train', '*.tfrecords'))
    test_tfrecords = glob.glob(os.path.join(data_dir, 'test', '*.tfrecords'))

    print(f'Number of training tfrecord files : {len(train_tfrecords)}')
    print(f'Number of test tfrecord files : {len(test_tfrecords)}')
    print(f'Total : {len(train_tfrecords) + len(test_tfrecords)}')

    train_dataset = build_dataset(train_tfrecords)
    test_dataset = build_dataset(test_tfrecords)

    train_dataset = train_dataset.map(lambda graph, img, c: (graph,))
    test_dataset = test_dataset.map(lambda graph, img, c: (graph,))

    train_dataset = batch_dataset_set_graph_tuples(all_graphs_same_size=True, dataset=train_dataset, batch_size=2)
    test_dataset = batch_dataset_set_graph_tuples(all_graphs_same_size=True, dataset=test_dataset, batch_size=2)

    model = GraphAutoEncoder(mlp_size=16,
                             cluster_encoded_size=10,
                             image_encoded_size=64,
                             num_heads=10,
                             kernel_size=4,
                             image_feature_size=16,
                             core_steps=10, name=None)

    learning_rate = 1e-5
    opt = snt.optimizers.Adam(learning_rate)

    def loss(model_outputs, batch):
        graph = batch
        decoded_graph = model_outputs
        return tf.reduce_mean((graph.nodes - decoded_graph.nodes) ** 2)

    train_one_epoch = TrainOneEpoch(model, loss, opt, strategy=None)

    log_dir = build_log_dir('test_log_dir', config)
    checkpoint_dir = build_checkpoint_dir('test_checkpointing', config)

    vanilla_training_loop(train_one_epoch=train_one_epoch,
                          training_dataset=train_dataset,
                          test_dataset=test_dataset,
                          num_epochs=20,
                          early_stop_patience=5,
                          checkpoint_dir=checkpoint_dir,
                          log_dir=log_dir,
                          debug=False)

def main(data_dir, config):
    # train_autoencoder(data_dir)
    train_ae_3d(data_dir, config)


if __name__ == '__main__':
    tfrec_base_dir = '/home/s2675544/data/tf_records'
    tfrec_dir = os.path.join(tfrec_base_dir, 'snap_128_tf_records')

    learning_rate = 1e-5
    mlp_size = 16
    cluster_encoded_size = 10
    image_encoded_size = 64
    core_steps = 28
    num_heads = 1

    config = dict(model_type='model1',
                  model_parameters=dict(mlp_size=mlp_size,
                                        cluster_encoded_size=cluster_encoded_size,
                                        image_encoded_size=image_encoded_size,
                                        core_steps=core_steps,
                                        num_heads=num_heads),
                  optimizer_parameters=dict(learning_rate=learning_rate, opt_type='adam'),
                  loss_parameters=dict())
    main(tfrec_dir, config)

import sys

sys.path.insert(1, '/home/s2675544/git/neural_deprojection/')

from neural_deprojection.graph_net_utils import vanilla_training_loop, TrainOneEpoch, AbstractModule, get_distribution_strategy, build_log_dir, build_checkpoint_dir
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
# from typing import Callable, Iterable, Optional, Text
# from sonnet.src import initializers
# from sonnet.src import linear
# import matplotlib.pyplot as plt
# import networkx as nx
# import numpy as np


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

    def _build(self, input_graph, num_processing_steps):
        latent_graph = self._encoder(input_graph)
        latent_graph0 = latent_graph
        # output_ops = []
        for _ in range(num_processing_steps):
            # core_input = concat([latent_graph0, latent_graph], axis=1)
            latent_graph = self._core(latent_graph0, latent_graph)
            # decoded_op = self._decoder(latent_graph)
            # output_ops.append(decoded_op)
        return self._decoder(latent_graph)

class CoreNetwork(AbstractModule):
    """
    Core network which can be used in the EncodeProcessDecode network. Consists of a (full) graph network block
    and a self attention block.
    """

    def __init__(self,
                 node_size,
                 name='core_network'):
        super(CoreNetwork, self).__init__(name=name)
        # self.graph_network = GraphNetwork(edge_model_fn=lambda: snt.nets.MLP([24, 16], activate_final=True),
        #                                   node_model_fn=lambda: snt.nets.MLP([32, node_size], activate_final=True),
        #                                   global_model_fn=lambda: snt.nets.MLP([32, 16], activate_final=True),
        #                                   reducer=tf.math.unsorted_segment_mean,
        #                                   edge_block_opt=dict(use_edges=False, use_globals=False),
        #                                   node_block_opt=dict(use_globals=False),
        #                                   global_block_opt=dict(use_globals=False))
        self.graph_network = RelationNetwork(lambda: snt.nets.MLP([32, 16], activate_final=True),
                                       lambda: snt.nets.MLP([32, 16], activate_final=True))
        self.self_attention = SelfAttention()

    def _build(self, core_input, latent_input):
        latent = self.graph_network(latent_input)
        attended_latent = self.self_attention(latent.nodes, core_input.nodes, core_input.nodes, latent)
        return attended_latent


class Autoencoder(AbstractModule):
    def __init__(self, kernel_size=4, name=None):
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
        self.epd_graph = EncodeProcessDecode(encoder=GraphIndependent(),
                                             core=CoreNetwork(node_size=10),
                                             decoder=GraphIndependent())
        self.epd_image = EncodeProcessDecode(encoder=GraphIndependent(),
                                             core=CoreNetwork(node_size=64),
                                             decoder=GraphIndependent())

        # Load the autoencoder model from checkpoint
        pretrained_auto_encoder = Autoencoder(kernel_size=kernel_size)

        checkpoint_dir = '/home/s2675544/git/neural_deprojection/neural_deprojection/models/identify_medium_GCD/autoencoder_checkpointing'
        encoder_decoder_cp = tf.train.Checkpoint(encoder=pretrained_auto_encoder.encoder, decoder=pretrained_auto_encoder.decoder)
        model_cp = tf.train.Checkpoint(_model=encoder_decoder_cp)
        checkpoint = tf.train.Checkpoint(module=model_cp)
        status = tf.train.latest_checkpoint(checkpoint_dir)
        checkpoint.restore(status).expect_partial()

        self.auto_encoder = pretrained_auto_encoder

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

        # The encoded cluster graph has globals which can be compared against the encoded image graph
        encoded_graph = self.epd_graph(graph, 20)

        # Add an extra dimension to the image (tf.summary expects a Tensor of rank 4)
        img = img[None, ...]

        print("IMG SHAPE:", img.shape)
        print("IMG MIN MAX:", tf.math.reduce_min(img), tf.math.reduce_max(img))

        img_before_cnn = (img - tf.reduce_min(img)) / \
                         (tf.reduce_max(img) - tf.reduce_min(img))
        tf.summary.image(f'img_before_cnn', img_before_cnn, step=self.step)

        # Smooth the image and use the encoder from the autoencoder to reduce the dimensionality of the image
        # The autoencoder was trained on images that were smoothed in the same way
        img = gaussian_filter2d(img, filter_shape=[6, 6])
        img = self.auto_encoder.encoder(img)

        # Prevent the autoencoder from learning
        try:
            for variable in self.auto_encoder.encoder.trainable_variables:
                variable._trainable = False
            for variable in self.auto_encoder.decoder.trainable_variables:
                variable._trainable = False
        except:
            pass

        print("IMG SHAPE AFTER CNN:", img.shape)
        print("IMG MIN MAX AFTER CNN:", tf.math.reduce_min(img), tf.math.reduce_max(img))

        img_after_cnn = (img - tf.reduce_min(img)) / \
                        (tf.reduce_max(img) - tf.reduce_min(img))
        tf.summary.image(f'img_after_cnn', tf.transpose(img_after_cnn, [3, 1, 2, 0]), step=self.step)

        decoded_img = self.auto_encoder.decoder(img)
        decoded_img = (decoded_img - tf.reduce_min(decoded_img)) / \
                                (tf.reduce_max(decoded_img) - tf.reduce_min(decoded_img))
        tf.summary.image(f'decoded_img', decoded_img, step=self.step)

        # Reshape the encoded image so it can be used for the nodes
        img_nodes = tf.reshape(img, (-1, self.image_feature_size))

        # Create a graph that has a node for every encoded pixel. The features of each node
        # are the channels of the corresponding pixel. Then connect each node with every other
        # node.
        img_graph = GraphsTuple(nodes=img_nodes,
                            edges=None,
                            globals=None,
                            receivers=None,
                            senders=None,
                            n_node=tf.shape(img_nodes)[0:1],
                            n_edge=tf.constant([0]))
        connected_graph = fully_connect_graph_dynamic(img_graph)

        # The encoded image graph has globals which can be compared against the encoded cluster graph
        encoded_img = self.epd_image(connected_graph, 1)

        # Compare the globals from the encoded cluster graph and encoded image graph
        # to estimate the similarity between the input graph and input image
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
                          early_stop_patience=5,
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
                          num_epochs=1,
                          early_stop_patience=3,
                          checkpoint_dir=checkpoint_dir,
                          log_dir=log_dir,
                          debug=False)

def main(data_dir, config):
    # train_autoencoder(data_dir)
    train_identify_medium(data_dir, config)


if __name__ == '__main__':
    tfrec_base_dir = '/home/s2675544/data/tf_records'
    tfrec_dir = os.path.join(tfrec_base_dir, 'snap_128_tf_records')

    learning_rate = 1e-4
    kernel_size = 4
    mlp_layers = 2
    image_feature_size = 64
    # conv_layers = 6
    mlp_layer_nodes = 32

    config = dict(model_type='model1',
                  model_parameters=dict(mlp_layers=mlp_layers,
                                        mlp_layer_nodes=mlp_layer_nodes,
                                        kernel_size=kernel_size,
                                        image_feature_size=image_feature_size),
                  optimizer_parameters=dict(learning_rate=learning_rate, opt_type='adam'),
                  loss_parameters=dict())
    main(tfrec_dir, config)

import tensorflow as tf
import sonnet as snt
import numpy as np
from graph_nets import blocks
from tensorflow_addons.image import gaussian_filter2d
from graph_nets.graphs import GraphsTuple
from graph_nets.modules import _unsorted_segment_softmax, _received_edges_normalizer, GraphIndependent, SelfAttention, GraphNetwork
from graph_nets.utils_tf import fully_connect_graph_dynamic, concat
from sonnet.src import utils, once
from neural_deprojection.graph_net_utils import AbstractModule


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

    def _build(self, input_graph, num_processing_steps):
        latent_graph = self._encoder(input_graph)
        # for _ in range(num_processing_steps):
        #     latent_graph = self._core(latent_graph)

        # state = (counter, latent_graph)
        _, latent_graph = tf.while_loop(cond=lambda const, state: const < num_processing_steps,
                      body=lambda const, state: (const+1, self._core(state)),
                      loop_vars=(tf.constant(0), latent_graph))

        return self._decoder(latent_graph)


class CoreNetwork(AbstractModule):
    """
    Core network which can be used in the EncodeProcessDecode network. Consists of a (full) graph network block
    and a self attention block.
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

    def _build(self, latent):
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

    def _build(self, input_graph):
        latent = self.node_block(input_graph)
        output = self.relation_network(latent)
        return output


class AutoEncoder(AbstractModule):
    def __init__(self, kernel_size=4, name=None):
        super(AutoEncoder, self).__init__(name=name)
        self.encoder = snt.Sequential([snt.Conv2D(4, kernel_size, stride=4, padding='SAME'), tf.nn.relu,
                                       snt.Conv2D(16, kernel_size, stride=4, padding='SAME'), tf.nn.relu,
                                       snt.Conv2D(64, kernel_size, stride=4, padding='SAME'), tf.nn.relu])

        self.decoder = snt.Sequential([snt.Conv2DTranspose(64, kernel_size, stride=4, padding='SAME'), tf.nn.relu,
                                       snt.Conv2DTranspose(16, kernel_size, stride=4, padding='SAME'), tf.nn.relu,
                                       snt.Conv2DTranspose(4, kernel_size, stride=4, padding='SAME'), tf.nn.relu,
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

    def __init__(self,
                 mlp_size=16,
                 cluster_encoded_size=10,
                 image_encoded_size=64,
                 num_heads=10,
                 kernel_size=4,
                 image_feature_size=16,
                 core_steps=10, name=None):
        super(Model, self).__init__(name=name)
        self.epd_graph = EncodeProcessDecode(encoder=EncoderNetwork(edge_model_fn=lambda: snt.nets.MLP([mlp_size], activate_final=True),
                                                                    node_model_fn=lambda: snt.Linear(cluster_encoded_size),
                                                                    global_model_fn=lambda: snt.nets.MLP([mlp_size], activate_final=True)),
                                             core=CoreNetwork(num_heads=num_heads,
                                                              multi_head_output_size=cluster_encoded_size,
                                                              input_node_size=cluster_encoded_size),
                                             decoder=EncoderNetwork(edge_model_fn=lambda: snt.nets.MLP([mlp_size], activate_final=True),
                                                                    node_model_fn=lambda: snt.Linear(cluster_encoded_size),
                                                                    global_model_fn=lambda: snt.nets.MLP([32, 32, 256], activate_final=True)))

        self.epd_image = EncodeProcessDecode(encoder=EncoderNetwork(edge_model_fn=lambda: snt.nets.MLP([mlp_size], activate_final=True),
                                                                    node_model_fn=lambda: snt.Linear(image_encoded_size),
                                                                    global_model_fn=lambda: snt.nets.MLP([mlp_size], activate_final=True)),
                                             core=CoreNetwork(num_heads=num_heads,
                                                              multi_head_output_size=image_encoded_size,
                                                              input_node_size=image_encoded_size),
                                             decoder=EncoderNetwork(edge_model_fn=lambda: snt.nets.MLP([mlp_size], activate_final=True),
                                                                    node_model_fn=lambda: snt.Linear(image_encoded_size),
                                                                    global_model_fn=lambda: snt.nets.MLP([32, 32, 256], activate_final=True)))

        # Load the autoencoder model from checkpoint
        pretrained_auto_encoder = AutoEncoder(kernel_size=kernel_size)
        checkpoint_dir = '/home/s2675544/git/neural_deprojection/neural_deprojection/models/identify_medium_GCD/autoencoder_checkpointing'
        encoder_decoder_cp = tf.train.Checkpoint(encoder=pretrained_auto_encoder.encoder, decoder=pretrained_auto_encoder.decoder)
        model_cp = tf.train.Checkpoint(_model=encoder_decoder_cp)
        checkpoint = tf.train.Checkpoint(module=model_cp)
        status = tf.train.latest_checkpoint(checkpoint_dir)

        checkpoint.restore(status).expect_partial()

        self.auto_encoder = pretrained_auto_encoder

        self.compare = snt.nets.MLP([32, 1])
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
        (graph, img, c) = batch
        del c

        print(graph.nodes.shape)

        # The encoded cluster graph has globals which can be compared against the encoded image graph
        encoded_graph = self.epd_graph(graph, self._core_steps)

        # # Add an extra dimension to the image (tf.summary expects a Tensor of rank 4)
        # img = img[None, ...]

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
        tf.summary.image(f'quantized_img', img_after_cnn[:, :, :, np.random.randint(low=0, high=64)][:, :, :, None],
                         step=self.step)

        decoded_img = self.auto_encoder.decoder(img)
        decoded_img = (decoded_img - tf.reduce_min(decoded_img)) / \
                      (tf.reduce_max(decoded_img) - tf.reduce_min(decoded_img))
        tf.summary.image(f'decoded_img', decoded_img, step=self.step)

        # Reshape the encoded image so it can be used for the nodes
        img_nodes = tf.reshape(img, (-1, self.image_feature_size))
        print(img_nodes.shape)

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
        print(encoded_img.globals.shape)
        print(encoded_graph.globals.shape)
        distance = self.compare(tf.concat([encoded_graph.globals, encoded_img.globals], axis=1)) + self.compare(tf.concat([encoded_img.globals, encoded_graph.globals], axis=1))
        return distance


def graph_tuple_to_feature(graph: GraphsTuple, name=''):
    return {
        f'{name}_nodes': tf.train.Feature(
            bytes_list=tf.train.BytesList(value=[tf.io.serialize_tensor(tf.cast(graph.nodes, tf.float32)).numpy()])),
        f'{name}_edges': tf.train.Feature(
            bytes_list=tf.train.BytesList(value=[tf.io.serialize_tensor(tf.cast(graph.edges, tf.float32)).numpy()])),
        f'{name}_senders': tf.train.Feature(
            bytes_list=tf.train.BytesList(value=[tf.io.serialize_tensor(tf.cast(graph.senders, tf.int64)).numpy()])),
        f'{name}_receivers': tf.train.Feature(
            bytes_list=tf.train.BytesList(value=[tf.io.serialize_tensor(tf.cast(graph.receivers, tf.int64)).numpy()]))}


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
    receivers = tf.io.parse_tensor(parsed_example['graph_receivers'], tf.int32)
    receivers.set_shape([None])
    senders = tf.io.parse_tensor(parsed_example['graph_senders'], tf.int32)
    senders.set_shape([None])

    n_node = tf.shape(graph_nodes)[0:1]
    n_edge = tf.shape(graph_edges)[0:1]

    # graph = GraphsTuple(nodes=graph_nodes,
    #                     edges=graph_edges,
    #                     globals=tf.zeros([1]),
    #                     receivers=receivers,
    #                     senders=senders,
    #                     n_node=n_node,
    #                     n_edge=n_edge)

    graph_data_dict = dict(nodes=graph_nodes,
                           edges=tf.zeros((n_edge[0], 1)),
                           globals=tf.zeros([1]),
                           receivers=receivers,
                           senders=senders,
                           n_node=n_node,
                           n_edge=n_edge)

    return (graph_data_dict, image, cluster_idx, projection_idx, vprime)
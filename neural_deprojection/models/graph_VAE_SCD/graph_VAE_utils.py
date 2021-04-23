import sys

sys.path.insert(1, '/data/s1825216/git/neural_deprojection/')

from neural_deprojection.graph_net_utils import AbstractModule, \
    histogramdd
import tensorflow as tf
# import tensorflow_addons as tfa

from graph_nets import blocks
import sonnet as snt
from graph_nets.modules import SelfAttention
from sonnet.src import utils, once


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

# TODO: give option to feed position in the core network
class EncodeProcessDecode_E(AbstractModule):
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
                 name="EncodeProcessDecode_E"):
        super(EncodeProcessDecode_E, self).__init__(name=name)
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


class EncodeProcessDecode_D(AbstractModule):
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
                 name="EncodeProcessDecode_D"):
        super(EncodeProcessDecode_D, self).__init__(name=name)
        self._encoder = encoder
        self._core = core
        self._decoder = decoder

    def _build(self, input_graph, num_processing_steps, positions):
        latent_graph = self._encoder(input_graph, positions)
        _, latent_graph = tf.while_loop(cond=lambda const, state: const < num_processing_steps,
                      body=lambda const, state: (const+1, self._core(state, positions)),
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
            prepend_nodes = tf.concat([positions, output_graph.nodes[:, 3:]], axis=1)
            output_graph = output_graph.replace(nodes=prepend_nodes)
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
            latent = latent.replace(nodes=prepend_nodes)

        output = self.relation_network(latent)
        return output


class DecoderNetwork(AbstractModule):
    """
    Encoder network that updates the graph to viable input for the Core network.
    Contains a node block to update the edges and a relation network to generate edges and globals.
    """

    def __init__(self,
                 node_model_fn,
                 name=None):
        super(DecoderNetwork, self).__init__(name=name)
        self.node_block = blocks.NodeBlock(node_model_fn,
                                           use_received_edges=False,
                                           use_sent_edges=False,
                                           use_nodes=False,
                                           use_globals=True)

    def _build(self, input_graph, positions):
        output = self.node_block(input_graph)
        output = output._replace(edges=tf.constant(1.))
        if positions is not None:
            prepend_nodes = tf.concat([positions, output.nodes[:, 3:]], axis=1)
            output = output.replace(nodes=prepend_nodes)
        return output


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
                 cluster_encoded_size=11,
                 num_heads=10,
                 core_steps=10, name=None):
        super(Model, self).__init__(name=name)

        self.epd_encoder = EncodeProcessDecode_E(encoder=EncoderNetwork(edge_model_fn=lambda: snt.nets.MLP([mlp_size], activate_final=True, activation=tf.nn.leaky_relu),
                                                                      node_model_fn=lambda: snt.Linear(cluster_encoded_size),
                                                                      global_model_fn=lambda: snt.nets.MLP([mlp_size], activate_final=True, activation=tf.nn.leaky_relu)),
                                               core=CoreNetwork(num_heads=num_heads,
                                                                multi_head_output_size=cluster_encoded_size,
                                                                input_node_size=cluster_encoded_size),
                                               decoder=EncoderNetwork(edge_model_fn=lambda: snt.nets.MLP([mlp_size], activate_final=True, activation=tf.nn.leaky_relu),
                                                                      node_model_fn=lambda: snt.Linear(cluster_encoded_size),
                                                                      global_model_fn=lambda: snt.nets.MLP([32, 32, 64], activate_final=True, activation=tf.nn.leaky_relu)))

        self.epd_decoder = EncodeProcessDecode_D(encoder=DecoderNetwork(node_model_fn=lambda: snt.nets.MLP([32, 32, cluster_encoded_size], activate_final=True, activation=tf.nn.leaky_relu)),
                                               core=CoreNetwork(num_heads=num_heads,
                                                                multi_head_output_size=cluster_encoded_size,
                                                                input_node_size=cluster_encoded_size),
                                               decoder=snt.Sequential([RelationNetwork(edge_model_fn=lambda: snt.nets.MLP([mlp_size], activate_final=True, activation=tf.nn.leaky_relu),
                                                                                       global_model_fn=lambda: snt.nets.MLP([mlp_size], activate_final=True, activation=tf.nn.leaky_relu)),
                                                                       blocks.NodeBlock(
                                                                           node_model_fn=lambda: snt.nets.MLP(
                                                                               [cluster_encoded_size-3], activate_final=True, activation=tf.nn.leaky_relu),
                                                                           use_received_edges=True,
                                                                           use_sent_edges=True,
                                                                           use_nodes=True,
                                                                           use_globals=True)
                                                                       ])
                                               )

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

        # del img
        # del c

        positions = graph.nodes[:, :3]


        for i in range(3, 11):
            image_before, _ = histogramdd(positions[:, :2], bins=50, weights=graph.nodes[:, i])
            image_before -= tf.reduce_min(image_before)
            image_before /= tf.reduce_max(image_before)
            tf.summary.image(f"{i}_xy_image_before", image_before[None, :, :, None], step=self.step)

            tf.summary.scalar(f"properties{i}_std_before", tf.math.reduce_std(graph.nodes[:,i]), step=self.step)

        encoded_graph = self.epd_encoder(graph, self._core_steps, positions)
        encoded_graph = encoded_graph._replace(nodes=None, edges=None)   # only pass through globals for sure
        decoded_graph = self.epd_decoder(encoded_graph, self._core_steps, positions)

        for i in range(8):
            image_after, _ = histogramdd(positions[:, :2], bins=50, weights=decoded_graph.nodes[:, i])
            image_after -= tf.reduce_min(image_after)
            image_after /= tf.reduce_max(image_after)
            tf.summary.image(f"{i+3}_xy_image_after", image_after[None, :, :, None], step=self.step)
            tf.summary.scalar(f"properties{i+3}_std_after", tf.math.reduce_std(decoded_graph.nodes[:,i]), step=self.step)

        return decoded_graph



import sys

sys.path.insert(1, '/data/s1825216/git/neural_deprojection/')

from neural_deprojection.graph_net_utils import AbstractModule, \
    histogramdd, efficient_nn_index
from neural_deprojection.graph_net_utils import AbstractModule, gaussian_loss_function, \
    reconstruct_fields_from_gaussians

import tensorflow as tf
# import tensorflow_addons as tfa

from graph_nets import blocks
import sonnet as snt
from graph_nets.modules import SelfAttention
from sonnet.src import utils, once
from tensorflow_probability.python.math.psd_kernels.internal import util
from graph_nets.utils_tf import fully_connect_graph_static, fully_connect_graph_dynamic, concat
from graph_nets.utils_np import graphs_tuple_to_networkxs, networkxs_to_graphs_tuple, get_graph
import numpy as np
import networkx as nx
from scipy.spatial.ckdtree import cKDTree
import time
from graph_nets.graphs import GraphsTuple
import tensorflow_probability as tfp


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
        output = self.node_block(input_graph.replace(n_node=tf.constant([positions.shape[0]], dtype=tf.int32)))
        output = output._replace(edges=tf.constant(1.))
        if positions is not None:
            prepend_nodes = tf.concat([positions, output.nodes[:, 3:]], axis=1)
            output = output.replace(nodes=prepend_nodes)
        return output


def nearest_neighbours_connected_graph(virtual_positions, k):
    kdtree = cKDTree(virtual_positions)
    dist, idx = kdtree.query(virtual_positions, k=k + 1)
    receivers = idx[:, 1:]  # N,k
    senders = np.arange(virtual_positions.shape[0])  # N
    senders = np.tile(senders[:, None], [1, k])  # N,k

    receivers = receivers.flatten()
    senders = senders.flatten()

    graph_nodes = tf.convert_to_tensor(virtual_positions, tf.float32)
    graph_nodes.set_shape([None, 3])
    receivers = tf.convert_to_tensor(receivers, tf.int32)
    receivers.set_shape([None])
    senders = tf.convert_to_tensor(senders, tf.int32)
    senders.set_shape([None])
    n_node = tf.shape(graph_nodes)[0:1]
    n_edge = tf.shape(senders)[0:1]

    graph_data_dict = dict(nodes=graph_nodes,
                           edges=tf.zeros((n_edge[0], 1)),
                           globals=tf.zeros([1]),
                           receivers=receivers,
                           senders=senders,
                           n_node=n_node,
                           n_edge=n_edge)

    return GraphsTuple(**graph_data_dict)


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
                 activation='leaky_relu',
                 mlp_size=16,
                 cluster_encoded_size=11,
                 num_heads=10,
                 core_steps=10, name=None):
        super(Model, self).__init__(name=name)

        if activation == 'leaky_relu':
            self.activation = tf.nn.leaky_relu
        elif activation == 'relu':
            self.activation = tf.nn.relu
        else:
            self.activation = tf.nn.relu

        self.epd_encoder = EncodeProcessDecode_E(encoder=EncoderNetwork(edge_model_fn=lambda: snt.nets.MLP([mlp_size], activate_final=True, activation=self.activation),
                                                                      node_model_fn=lambda: snt.Linear(cluster_encoded_size),
                                                                      global_model_fn=lambda: snt.nets.MLP([mlp_size], activate_final=True, activation=self.activation)),
                                               core=CoreNetwork(num_heads=num_heads,
                                                                multi_head_output_size=cluster_encoded_size,
                                                                input_node_size=cluster_encoded_size),
                                               decoder=EncoderNetwork(edge_model_fn=lambda: snt.nets.MLP([mlp_size], activate_final=True, activation=self.activation),
                                                                      node_model_fn=lambda: snt.Linear(cluster_encoded_size),
                                                                      global_model_fn=lambda: snt.nets.MLP([32, 32, 64], activate_final=True, activation=self.activation)))

        self.epd_decoder = EncodeProcessDecode_D(encoder=DecoderNetwork(node_model_fn=lambda: snt.nets.MLP([32, 32, cluster_encoded_size], activate_final=True, activation=self.activation)),
                                               core=CoreNetwork(num_heads=num_heads,
                                                                multi_head_output_size=cluster_encoded_size,
                                                                input_node_size=cluster_encoded_size),
                                               decoder=snt.Sequential([RelationNetwork(edge_model_fn=lambda: snt.nets.MLP([mlp_size], activate_final=True, activation=self.activation),
                                                                                       global_model_fn=lambda: snt.nets.MLP([mlp_size], activate_final=True, activation=self.activation)),
                                                                       blocks.NodeBlock(
                                                                           node_model_fn=lambda: snt.nets.MLP(
                                                                               [cluster_encoded_size-3], activate_final=True, activation=self.activation),
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

        t0 = time.time()
        encoded_graph = self.epd_encoder(graph, self._core_steps, positions)
        encoded_graph = encoded_graph._replace(nodes=None, edges=None, receivers=None, senders=None)   # only pass through globals for sure
        # decoded_graph = self.epd_decoder(encoded_graph, self._core_steps, positions)
        t1 = time.time()
        print(f'encoder time {t1-t0} s')

        number_of_nodes = positions.shape[0]
        decode_positions = tf.random.uniform(shape=(number_of_nodes, 3),
                                             minval=tf.reduce_min(positions, axis=0),
                                             maxval=tf.reduce_max(positions, axis=0))

        t2 = time.time()
        print(f'decode pos time {t2 - t1} s')

        # encoded_graph = encoded_graph._replace(nodes=decode_positions)

        random_pos_graph = nearest_neighbours_connected_graph(decode_positions, 6)
        t3 = time.time()
        print(f'random pos time {t3 - t2} s')

        random_pos_graph = random_pos_graph._replace(nodes=None, edges=None, globals=encoded_graph.globals.numpy())

        t4 = time.time()
        print(f'replace pos time {t4 - t3} s')

        # encoded_graph = fully_connect_graph_static(encoded_graph)  # TODO: only works if batch_size=1, might need to use dynamic

        t4 = time.time()
        print(f'random pos time {t4 - t3} s')

        decoded_graph = self.epd_decoder(random_pos_graph, self._core_steps, decode_positions)

        t5 = time.time()
        print(f'decoder time {t5 - t4} s')

        nn_index = efficient_nn_index(decode_positions, positions)

        t6 = time.time()
        print(f'nn time {t6 - t5} s')

        for i in range(8):
            image_after, _ = histogramdd(decode_positions[:, :2], bins=50, weights=decoded_graph.nodes[:, i])
            image_after -= tf.reduce_min(image_after)
            image_after /= tf.reduce_max(image_after)
            tf.summary.image(f"{i+3}_xy_image_after", image_after[None, :, :, None], step=self.step)
            tf.summary.scalar(f"properties{i+3}_std_after", tf.math.reduce_std(decoded_graph.nodes[:,i]), step=self.step)

        return decoded_graph, nn_index


class DiscreteGraphVAE(AbstractModule):
    def __init__(self, encoder_fn: AbstractModule,
                 decode_fn: AbstractModule,
                 embedding_dim: int = 64,
                 num_embedding: int = 1024,
                 num_gaussian_components: int=128,
                 num_token_samples: int = 1,
                 num_properties: int = 10,
                 temperature: float = 50.,
                 beta: float = 1.,
                 encoder_kwargs: dict = None,
                 decode_kwargs: dict = None,
                 name=None):
        super(DiscreteGraphVAE, self).__init__(name=name)
        # (num_embedding, embedding_dim)

        self.temperature = temperature
        self.beta = beta

        self.embeddings = tf.Variable(initial_value=tf.random.truncated_normal((num_embedding, embedding_dim)),
                                      name='embeddings')
        self.encoder = encoder_fn(num_output=num_embedding, output_size=embedding_dim,
                                  **encoder_kwargs)
        self.decoder = decode_fn(num_output=num_gaussian_components, output_size=num_properties*10,
                                 **decode_kwargs)
        self.num_token_samples = num_token_samples
        self.num_properties = num_properties
        self.num_embedding = num_embedding

    # @tf.function(input_signature=tf.TensorSpec(shape=[None], dtype=tf.float32))  # what is the shape ???
    # def sample_encoder(self, graph):
    #     return self.encoder(graph)

    @tf.function(input_signature=[tf.TensorSpec([None,3], dtype=tf.float32),
                                  tf.TensorSpec([None,None], dtype=tf.float32),
                                  tf.TensorSpec([], dtype=tf.float32)])
    def sample_decoder(self, positions, logits, temperature):
        token_distribution = tfp.distributions.RelaxedOneHotCategorical(temperature, logits=logits)
        token_samples_onehot = token_distribution.sample((1,),
                                                         name='token_samples')
        token_sample_onehot = token_samples_onehot[0]#[n_node, num_embedding]
        token_sample = tf.matmul(token_sample_onehot, self.embeddings)  # [n_node, embedding_dim]
        n_node = tf.shape(token_sample)[0]
        latent_graph = GraphsTuple(nodes=token_sample,
                                   edges=None,
                                   globals=tf.constant([0.], dtype=tf.float32),
                                   senders=None,
                                   receivers=None,
                                   n_node=[n_node],
                                   n_edge=tf.constant([0], dtype=tf.int32))  # [n_node, embedding_dim]
        latent_graph = fully_connect_graph_dynamic(latent_graph)
        gaussian_tokens = self.decoder(latent_graph)  # nodes=[num_gaussian_components, component_dim]
        reconstructed_fields = reconstruct_fields_from_gaussians(gaussian_tokens, positions)
        return reconstructed_fields

    @property
    def step(self):
        if self._step is None:
            raise ValueError("Need to set step idx variable. model.step = epoch")
        return self._step

    @step.setter
    def step(self, value):
        self._step = value

    def _build(self, batch, **kwargs) -> dict:
        # graph, temperature, beta = batch
        graph = batch
        encoded_graph = self.encoder(graph)
        print('encoded_graph', encoded_graph)
        print(dir(encoded_graph.nodes))
        encoded_graph.replace(nodes=encoded_graph.nodes[10000:])
        n_node = encoded_graph.n_node
        # nodes = [n_node, num_embeddings]
        # node = [num_embeddings] -> log(p_i) = logits
        # -> [S, n_node, embedding_dim]
        logits = encoded_graph.nodes  # [n_node, num_embeddings]
        log_norm = tf.math.reduce_logsumexp(logits, axis=1)  # [n_node]
        token_distribution = tfp.distributions.RelaxedOneHotCategorical(self.temperature, logits=logits)
        token_samples_onehot = token_distribution.sample((self.num_token_samples,),
                                                         name='token_samples')  # [S, n_node, num_embeddings]

        def _single_decode(token_sample_onehot):
            """

            Args:
                token_sample: [n_node, embedding_dim]

            Returns:
                log_likelihood: scalar
                kl_term: scalar
            """
            token_sample = tf.matmul(token_sample_onehot, self.embeddings)  # [n_node, embedding_dim]  # = z ~ q(z|x)
            latent_graph = GraphsTuple(nodes=token_sample,
                                       edges=None,
                                       globals=tf.constant([0.], dtype=tf.float32),
                                       senders=None,
                                       receivers=None,
                                       n_node=n_node,
                                       n_edge=tf.constant([0], dtype=tf.int32))  # [n_node, embedding_dim]
            print('latent_graph', latent_graph)
            latent_graph = fully_connect_graph_dynamic(latent_graph)
            gaussian_tokens = self.decoder(latent_graph)  # nodes=[num_gaussian_components, component_dim]
            _, log_likelihood = gaussian_loss_function(gaussian_tokens.nodes, graph)
            # [n_node, num_embeddings].[n_node, num_embeddings]
            sum_selected_logits = tf.math.reduce_sum(token_sample_onehot * logits, axis=1)  # [n_node]
            kl_term = sum_selected_logits - tf.cast(self.num_embedding, tf.float32) * tf.cast(log_norm, tf.float32) + \
                      tf.cast(self.num_embedding, tf.float32) * tf.math.log(tf.cast(self.num_embedding, tf.float32))  # [n_node]
            kl_term = self.beta * tf.reduce_mean(kl_term)
            return log_likelihood, kl_term

        print('token_samples_onehot',token_samples_onehot)

        log_likelihood_samples, kl_term_samples = _single_decode(token_samples_onehot[0])  # tf.vectorized_map(_single_decode, token_samples_onehot)  # [S],[S]

        # good metric = average entropy of embedding usage! The more precisely embeddings are selected the lower the entropy.

        log_prob_tokens = logits - log_norm[:, None]#num_tokens, num_embeddings
        entropy = -tf.reduce_sum(log_prob_tokens * tf.math.exp(log_prob_tokens), axis=1)#num_tokens
        perplexity = 2.**(-entropy/tf.math.log(2.))
        mean_perplexity = tf.reduce_mean(perplexity)

        var_exp = tf.reduce_mean(log_likelihood_samples)
        tf.summary.scalar('var_exp', var_exp, step=self._step)

        kl_term=tf.reduce_mean(kl_term_samples)
        tf.summary.scalar('kl_term', kl_term, step=self._step)

        tf.summary.scalar('mean_perplexity', mean_perplexity, step=self._step)

        return dict(loss=tf.reduce_mean(log_likelihood_samples - kl_term_samples),
                    var_exp=var_exp,
                    kl_term=tf.reduce_mean(kl_term_samples),
                    mean_perplexity=mean_perplexity)


class GraphMappingNetwork(AbstractModule):
    """
    Encoder network that updates the graph to viable input for the DiscreteGraphVAE network.
    """

    def __init__(self,
                 num_output: int,
                 output_size: int,
                 node_size: int = 4,
                 edge_size: int = 4,
                 starting_global_size: int = 10,
                 inter_graph_connect_prob: float = 0.01,
                 crossing_steps: int = 4,
                 reducer=tf.math.unsorted_segment_mean,
                 properties_size=10,
                 name=None):
        super(GraphMappingNetwork, self).__init__(name=name)
        self.num_output = num_output
        self.output_size = output_size
        self.crossing_steps=crossing_steps
        self.empty_node_variable = tf.Variable(initial_value=tf.random.truncated_normal((node_size,)),
                                               name='empty_token_node')

        # values for different kinds of edges in the graph, which will be learned
        self.intra_graph_edge_variable = tf.Variable(initial_value=tf.random.truncated_normal((edge_size,)),
                                                     name='intra_graph_edge_var')
        self.intra_token_graph_edge_variable = tf.Variable(initial_value=tf.random.truncated_normal((edge_size,)),
                                                           name='intra_token_graph_edge_var')
        self.inter_graph_edge_variable = tf.Variable(initial_value=tf.random.truncated_normal((edge_size,)),
                                                     name='inter_graph_edge_var')
        self.starting_global_variable = tf.Variable(initial_value=tf.random.truncated_normal((starting_global_size,)),
                                                    name='starting_global_var')

        self.inter_graph_connect_prob = inter_graph_connect_prob

        self.projection_node_block = blocks.NodeBlock(lambda: snt.Linear(node_size, name='project'),
                                                      use_received_edges=False,
                                                      use_sent_edges=False,
                                                      use_nodes=True,
                                                      use_globals=False)

        node_model_fn = lambda: snt.nets.MLP([node_size, node_size], activate_final=True, activation=tf.nn.leaky_relu)
        edge_model_fn = lambda: snt.nets.MLP([edge_size, edge_size], activate_final=True, activation=tf.nn.leaky_relu)
        global_model_fn = lambda: snt.nets.MLP([starting_global_size, starting_global_size], activate_final=True,
                                               activation=tf.nn.leaky_relu)

        self.edge_block = blocks.EdgeBlock(edge_model_fn,
                                           use_edges=True,
                                           use_receiver_nodes=True,
                                           use_sender_nodes=True,
                                           use_globals=True)

        self.node_block = blocks.NodeBlock(node_model_fn,
                                           use_received_edges=True,
                                           use_sent_edges=True,
                                           use_nodes=True,
                                           use_globals=True)

        self.global_block = blocks.GlobalBlock(global_model_fn,
                                               use_edges=True,
                                               use_nodes=True,
                                               use_globals=True,
                                               edges_reducer=reducer)

        self.output_projection_node_block = blocks.NodeBlock(lambda: snt.Linear(self.output_size, name='project'),
                                                             use_received_edges=False,
                                                             use_sent_edges=False,
                                                             use_nodes=True,
                                                             use_globals=False)

    def _build(self, graph):
        n_edge = graph.n_edge[0]
        graph = graph.replace(edges=tf.tile(self.intra_graph_edge_variable[None, :], [n_edge, 1]))
        graph = self.projection_node_block(graph)  # [n_nodes, node_size]
        n_node = tf.shape(graph.nodes)[0]
        # create fully connected output token nodes
        token_start_nodes = tf.tile(self.empty_node_variable[None, :], [self.num_output, 1])
        graph.replace(n_node=tf.constant(n_node, dtype=tf.int32))
        token_graph = GraphsTuple(nodes=token_start_nodes,
                                  edges=None,
                                  globals=tf.constant([0.], dtype=tf.float32),
                                  senders=None,
                                  receivers=None,
                                  n_node=tf.constant([self.num_output], dtype=tf.int32),
                                  n_edge=tf.constant([0], dtype=tf.int32))
        token_graph = fully_connect_graph_static(token_graph)
        n_edge = token_graph.n_edge[0]
        token_graph = token_graph.replace(edges=tf.tile(self.intra_token_graph_edge_variable[None, :], [n_edge, 1]))
        concat_graph = concat([graph, token_graph], axis=0)  # n_node = [n_nodes, n_tokes]
        concat_graph = concat_graph.replace(n_node=tf.reduce_sum(concat_graph.n_node, keepdims=True),
                                            n_edge=tf.reduce_sum(concat_graph.n_edge, keepdims=True))  # n_node=[n_nodes+n_tokens]

        # add random edges between
        # choose random unique set of nodes in graph, choose random set of nodes in token_graph
        gumbel = -tf.math.log(-tf.math.log(tf.random.uniform((n_node,))))
        n_connect_edges = tf.cast(tf.multiply(tf.constant([self.inter_graph_connect_prob]), tf.cast(n_node, tf.float32)), tf.int32)
        _, graph_senders = tf.nn.top_k(gumbel, n_connect_edges[0])
        token_graph_receivers = n_node + tf.random.uniform(shape=n_connect_edges, minval=0, maxval=self.num_output,
                                                           dtype=tf.int32)
        senders = tf.concat([concat_graph.senders, graph_senders, token_graph_receivers],
                            axis=0)  # add bi-directional senders + receivers
        receivers = tf.concat([concat_graph.receivers, token_graph_receivers, graph_senders], axis=0)
        inter_edges = tf.tile(self.inter_graph_edge_variable[None, :], tf.concat([2 * n_connect_edges, tf.constant([1], dtype=tf.int32)], axis=0))  # 200 = 10000(n_nodes) * 0.01 * 2
        edges = tf.concat([concat_graph.edges, inter_edges], axis=0)
        concat_graph = concat_graph.replace(senders=senders, receivers=receivers, edges=edges,
                                            n_edge=concat_graph.n_edge[0] + 2 * n_connect_edges[0], # concat_graph.n_edge[0] + 2 * n_connect_edges
                                            globals=self.starting_global_variable[None, :])

        latent_graph = concat_graph
        print('concat_graph', concat_graph)
        for _ in range(
                self.crossing_steps):  # this would be that theoretical crossing time for information through the graph
            input_nodes = latent_graph.nodes
            latent_graph = self.edge_block(latent_graph)
            latent_graph = self.node_block(latent_graph)
            latent_graph = self.global_block(latent_graph)
            latent_graph = latent_graph.replace(nodes=latent_graph.nodes + input_nodes)  # residual connections

        latent_graph = latent_graph.replace(nodes=latent_graph.nodes[n_node:])
        output_graph = self.output_projection_node_block(latent_graph)

        return output_graph


class EncoderNetwork3D(GraphMappingNetwork):
    def __init__(self, num_output: int,
                 output_size: int,
                 inter_graph_connect_prob: float = 0.01,
                 reducer=tf.math.unsorted_segment_mean,
                 starting_global_size=4,
                 node_size=64,
                 edge_size=4,
                 crossing_steps=4,
                 name=None):
        super(EncoderNetwork3D, self).__init__(num_output=num_output,
                                               output_size=output_size,
                                               inter_graph_connect_prob=inter_graph_connect_prob,
                                               reducer=reducer,
                                               starting_global_size=starting_global_size,
                                               node_size=node_size,
                                               edge_size=edge_size,
                                               crossing_steps=crossing_steps,
                                               name=name)

class DecoderNetwork3D(GraphMappingNetwork):
    def __init__(self, num_output: int,
                 output_size: int,
                 inter_graph_connect_prob: float = 0.01,
                 reducer=tf.math.unsorted_segment_mean,
                 starting_global_size=4,
                 node_size=64,
                 edge_size=4,
                 crossing_steps=4,
                 name=None):
        super(DecoderNetwork3D, self).__init__(num_output=num_output,
                                               output_size=output_size,
                                               inter_graph_connect_prob=inter_graph_connect_prob,
                                               reducer=reducer,
                                               starting_global_size=starting_global_size,
                                               node_size=node_size,
                                               edge_size=edge_size,
                                               crossing_steps=crossing_steps,
                                               name=name)



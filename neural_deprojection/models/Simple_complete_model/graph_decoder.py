from graph_nets import blocks
import tensorflow as tf
import sonnet as snt
from graph_nets.graphs import GraphsTuple
from graph_nets import utils_tf
from neural_deprojection.graph_net_utils import AbstractModule, get_shape, graph_batch_reshape, graph_unbatch_reshape
import tensorflow_probability as tfp
from graph_nets.modules import SelfAttention
from sonnet.src import utils, once



def _create_autogressive_edges_from_nodes_dynamic(n_node, exclude_self_edges):
    """Creates complete edges for a graph with `n_node`.

    Args:
    n_node: (integer scalar `Tensor`) The number of nodes.
    exclude_self_edges: (bool) Excludes self-connected edges.

    Returns:
    A dict of RECEIVERS, SENDERS and N_EDGE data (`Tensor`s of rank 1).
    """

    rng = tf.range(n_node)

    if exclude_self_edges:
        ind = rng[:, None] > rng
        n_edge = n_node * (n_node - 1) // 2
    else:
        ind = rng[:, None] >= rng
        n_edge = n_node * (n_node - 1) // 2 + n_node

    indicies = tf.where(ind)
    receivers = indicies[:, 0]
    senders = indicies[:, 1]

    receivers = tf.reshape(tf.cast(receivers, tf.int32), [n_edge])
    senders = tf.reshape(tf.cast(senders, tf.int32), [n_edge])
    n_edge = tf.reshape(n_edge, [1])

    return {'receivers': receivers, 'senders': senders, 'n_edge': n_edge}


def test_autoregressive_connect_graph_dynamic():
    graphs = GraphsTuple(nodes=tf.range(20), n_node=tf.constant([12, 8]), n_edge=tf.constant([0, 0]),
                         edges=None, receivers=None, senders=None, globals=None)
    graphs = autoregressive_connect_graph_dynamic(graphs, exclude_self_edges=False)
    import networkx as nx
    G = nx.MultiDiGraph()
    for sender, receiver in zip(graphs.senders.numpy(), graphs.receivers.numpy()):
        G.add_edge(sender, receiver)
    nx.drawing.draw_circular(G)
    import pylab as plt
    plt.show()


def autoregressive_connect_graph_dynamic(graph,
                                         exclude_self_edges=False,
                                         name="autoregressive_connect_graph_dynamic"):
    """Adds edges to a graph by fully-connecting the nodes.

    This method does not require the number of nodes per graph to be constant,
    or to be known at graph building time.

    Args:
    graph: A `graphs.GraphsTuple` with `None` values for the edges, senders and
      receivers.
    exclude_self_edges (default=False): Excludes self-connected edges.
    name: (string, optional) A name for the operation.

    Returns:
    A `graphs.GraphsTuple` containing `Tensor`s with fully-connected edges.

    Raises:
    ValueError: if any of the `EDGES`, `RECEIVERS` or `SENDERS` field is not
      `None` in `graph`.
    """
    utils_tf._validate_edge_fields_are_all_none(graph)

    with tf.name_scope(name):
        def body(i, senders, receivers, n_edge):
            edges = _create_autogressive_edges_from_nodes_dynamic(graph.n_node[i],
                                                                  exclude_self_edges)
            return (i + 1, senders.write(i, edges['senders']),
                    receivers.write(i, edges['receivers']),
                    n_edge.write(i, edges['n_edge']))

        num_graphs = utils_tf.get_num_graphs(graph)
        loop_condition = lambda i, *_: tf.less(i, num_graphs)
        initial_loop_vars = [0] + [
            tf.TensorArray(dtype=tf.int32, size=num_graphs, infer_shape=False)
            for _ in range(3)  # senders, receivers, n_edge
        ]
        _, senders_array, receivers_array, n_edge_array = tf.while_loop(
            loop_condition, body, initial_loop_vars, back_prop=False)

        n_edge = n_edge_array.concat()
        offsets = utils_tf._compute_stacked_offsets(graph.n_node, n_edge)
        senders = senders_array.concat() + offsets
        receivers = receivers_array.concat() + offsets
        senders.set_shape(offsets.shape)
        receivers.set_shape(offsets.shape)

        receivers.set_shape([None])
        senders.set_shape([None])

        num_graphs = graph.n_node.get_shape().as_list()[0]
        n_edge.set_shape([num_graphs])

        return graph.replace(senders=senders, receivers=receivers, n_edge=n_edge)


### Use this one###
class GraphMappingNetwork(AbstractModule):
    """
    This autoregressive models the creation of a set of nodes based on some pre-existing set of tokens.
    This is done by adding a set of nodes to the set of tokens for each graph, and incrementally populating them.
    To do this, an artificial ordering must be imposed on the nodes.

    Args:
        num_output: number of new nodes to map to for each graph.
        node_size: size of the nodes during computation. Input nodes will be projected to this size.
        edge_size: size of edges during computation.
        global_size: size of global during computation.

    """

    def __init__(self,
                 num_output: int,
                 num_embedding: int,
                 multi_head_output_size: int,
                 num_heads: int = 4,
                 embedding_size: int = 4,
                 edge_size: int = 4,
                 global_size: int = 10,
                 name=None):
        super(GraphMappingNetwork, self).__init__(name=name)
        self.num_output = num_output
        self.embedding_size = embedding_size
        self.num_embedding = num_embedding
        self.edge_size = edge_size

        self.embeddings = tf.Variable(initial_value=tf.random.truncated_normal((num_embedding, embedding_size)))

        # starting value of nodes to be computed.
        self.starting_node_variable = tf.Variable(initial_value=tf.random.truncated_normal((num_output, embedding_size)),
                                                  name='starting_token_node')

        # starting value of globals
        self.starting_global_variable = tf.Variable(initial_value=tf.random.truncated_normal((global_size,)),
                                                    name='starting_global_var')

        self.projection_node_block = blocks.NodeBlock(lambda: snt.Linear(embedding_size, name='project'),
                                                      use_received_edges=False,
                                                      use_sent_edges=False,
                                                      use_nodes=True,
                                                      use_globals=False)

        node_model_fn = lambda: snt.nets.MLP([num_embedding, num_embedding], activate_final=True, activation=tf.nn.leaky_relu)
        edge_model_fn = lambda: snt.nets.MLP([edge_size, edge_size], activate_final=True, activation=tf.nn.leaky_relu)

        self.edge_block = blocks.EdgeBlock(edge_model_fn,
                                           use_edges=False,
                                           use_receiver_nodes=True,
                                           use_sender_nodes=True,
                                           use_globals=True)

        self.node_block = blocks.NodeBlock(node_model_fn,
                                           use_received_edges=True,
                                           use_sent_edges=True,
                                           use_nodes=True,
                                           use_globals=True)

        basis_weight_node_model_fn = lambda: snt.nets.MLP([num_embedding, 1], activate_final=True,
                                             activation=tf.nn.leaky_relu)
        basis_weight_edge_model_fn = lambda: snt.nets.MLP([edge_size, edge_size], activate_final=True, activation=tf.nn.leaky_relu)

        self.basis_weight_edge_block = blocks.EdgeBlock(basis_weight_edge_model_fn,
                                           use_edges=False,
                                           use_receiver_nodes=True,
                                           use_sender_nodes=True,
                                           use_globals=True)

        self.basis_weight_node_block = blocks.NodeBlock(basis_weight_node_model_fn,
                                           use_received_edges=True,
                                           use_sent_edges=True,
                                           use_nodes=True,
                                           use_globals=True)

        self.selfattention_core = CoreNetwork(num_heads=num_heads,
                                              multi_head_output_size=multi_head_output_size,
                                              input_node_size=embedding_size)

        self.selfattention_weights = CoreNetwork(num_heads=num_heads,
                                              multi_head_output_size=multi_head_output_size,
                                              input_node_size=embedding_size)



    def _build(self, graphs, temperature):
        """
        Adds another set of nodes to each graph. Autoregressively links all nodes in a graph.

        Args:
            graphs: batched GraphsTuple, node_shape = [n_graphs * num_input, input_embedding_size]
            temperature: scalar > 0
        Returns:
            #todo: give shapes to returns
            token_node
             kl_div
             token_3d_samples_onehot
             basis_weights

        """
        # give graphs edges and new node dimension (linear transformation)
        graphs = self.projection_node_block(graphs)  # nodes = [n_graphs * num_input, embedding_size]
        batched_graphs = graph_batch_reshape(graphs)  # nodes = [n_graphs, num_input, embedding_size]
        [n_graphs, n_node_per_graph_before_concat, _] = get_shape(batched_graphs.nodes)

        concat_nodes = tf.concat([batched_graphs.nodes, tf.tile(self.starting_node_variable[None, :], [n_graphs, 1, 1])],
                                 axis=-2)  # [n_graphs, num_input + num_output, embedding_size]
        batched_graphs = batched_graphs.replace(nodes=concat_nodes,
                                                globals=tf.tile(self.starting_global_variable[None, :], [n_graphs, 1]),
                                                n_node=tf.fill([n_graphs], n_node_per_graph_before_concat + self.num_output))
        concat_graphs = graph_unbatch_reshape(batched_graphs)  # [n_graphs * (num_input + num_output), embedding_size]

        # nodes, senders, receivers, globals
        concat_graphs = autoregressive_connect_graph_dynamic(concat_graphs)  # exclude self edges because 3d tokens orginally placeholder?

        # todo: this only works if exclude_self_edges=False
        n_edge = n_graphs * ((n_node_per_graph_before_concat + self.num_output) *
                             (n_node_per_graph_before_concat + self.num_output - 1) // 2 +
                             (n_node_per_graph_before_concat + self.num_output))

        latent_graphs = concat_graphs.replace(edges=tf.tile(tf.constant(self.edge_size * [0.])[None, :], [n_edge, 1]))
        latent_graphs.receivers.set_shape([n_edge])
        latent_graphs.senders.set_shape([n_edge])

        def _core(output_token_idx, latent_graphs, prev_kl_term, prev_token_3d_samples_onehot):
            batched_latent_graphs = graph_batch_reshape(latent_graphs)
            batched_input_nodes = batched_latent_graphs.nodes  # [n_graphs, num_input + num_output, embedding_size]

            # todo: use self-attention
            latent_graphs = self.selfattention_core(latent_graphs)
            # latent_graphs = self.edge_block(latent_graphs)     # also use node & edge blocks?
            # latent_graphs = self.node_block(latent_graphs)

            # batched_latent_graphs = graph_batch_reshape(latent_graphs)

            token_3d_logits = batched_latent_graphs.nodes[:, -self.num_output:, :]  # n_graphs, num_output, num_embedding
            reduce_logsumexp = tf.math.reduce_logsumexp(token_3d_logits, axis=-1)  # [n_graphs, num_output]
            reduce_logsumexp = tf.tile(reduce_logsumexp[..., None],
                                       [1, 1, self.num_embedding])  # [ n_graphs, num_output, num_embedding]
            token_3d_logits -= reduce_logsumexp

            token_distribution = tfp.distributions.RelaxedOneHotCategorical(temperature, logits=token_3d_logits)
            token_3d_samples_onehot = token_distribution.sample((1,),
                                                                name='token_samples')  # [1, n_graphs, num_output, num_embedding]
            token_3d_samples_onehot = token_3d_samples_onehot[0]  # [n_graphs, num_output, num_embedding]
            # token_3d_samples_max_index = tf.math.argmax(token_3d_logits, axis=-1, output_type=tf.int32)
            # token_3d_samples_onehot = tf.cast(tf.tile(tf.range(self.num_embedding)[None, None, :], [n_graphs, self.num_output, 1]) ==
            #                                   token_3d_samples_max_index[:,:,None], tf.float32)  # [n_graphs, num_output, num_embedding]
            token_3d_samples = tf.einsum('goe,ed->god', token_3d_samples_onehot,
                                         self.embeddings)  # [n_graphs, num_ouput, embedding_dim]
            _mask = tf.range(self.num_output) == output_token_idx  # [num_output]
            mask = tf.concat([tf.zeros(n_node_per_graph_before_concat, dtype=tf.bool), _mask], axis=0)  # num_input + num_output
            mask = tf.tile(mask[None, :, None], [n_graphs, 1, self.embedding_size])  # [n_graphs, num_input + num_output, embedding_size]

            kl_term = tf.reduce_sum((token_3d_samples_onehot * token_3d_logits), axis=-1)  # [n_graphs, num_output]
            kl_term = tf.reduce_sum(tf.cast(_mask, tf.float32) * kl_term, axis=-1)  # [n_graphs]
            kl_term += prev_kl_term

            # n_graphs, n_node+num_output, embedding_size
            output_nodes = tf.where(mask,
                                    tf.concat([tf.zeros([n_graphs, n_node_per_graph_before_concat, self.embedding_size]),
                                               token_3d_samples], axis=1),
                                    batched_input_nodes)
            batched_latent_graphs = batched_latent_graphs.replace(nodes=output_nodes)
            latent_graphs = graph_unbatch_reshape(batched_latent_graphs)

            return (output_token_idx + 1, latent_graphs, kl_term, token_3d_samples_onehot)

        _, latent_graphs, kl_div, token_3d_samples_onehot = tf.while_loop(
            cond=lambda output_token_idx, state, _, __: output_token_idx < self.num_output,
            body=lambda output_token_idx, state, prev_kl_term, prev_token_3d_samples_onehot: _core(output_token_idx, state, prev_kl_term, prev_token_3d_samples_onehot),
            loop_vars=(tf.constant([0]), latent_graphs, tf.zeros((n_graphs,), dtype=tf.float32), tf.zeros((n_graphs, self.num_output, self.num_embedding), dtype=tf.float32)))

        # compute weights for how much each basis function will contribute, forcing later ones to contribute less.
        #todo: use self-attention
        basis_weight_graphs = self.selfattention_weights(latent_graphs)
        basis_weight_graphs = self.basis_weight_node_block(self.basis_weight_edge_block(basis_weight_graphs))
        basis_weight_graphs = graph_batch_reshape(basis_weight_graphs)
        #[n_graphs, num_output]
        basis_weights = basis_weight_graphs.nodes[:,-self.num_output, 0]
        #make the weights shrink with increasing component
        basis_weights = tf.math.cumprod(tf.nn.sigmoid(basis_weights), axis=-1)

        latent_graphs = graph_batch_reshape(latent_graphs)
        token_nodes = latent_graphs.nodes[:, -self.num_output:, :]

        return token_nodes, kl_div, token_3d_samples_onehot, basis_weights


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
        return


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
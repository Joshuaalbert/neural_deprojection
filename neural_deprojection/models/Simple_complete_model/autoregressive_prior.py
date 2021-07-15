from graph_nets import blocks
import tensorflow as tf
import sonnet as snt
from graph_nets.graphs import GraphsTuple
from graph_nets import utils_tf
from neural_deprojection.graph_net_utils import AbstractModule, get_shape, graph_batch_reshape, graph_unbatch_reshape
from neural_deprojection.models.Simple_complete_model.autoencoder_3d import DiscreteVoxelsVAE
from neural_deprojection.models.Simple_complete_model.autoencoder_2d import DiscreteImageVAE
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
    graphs = GraphsTuple(nodes=tf.range(6), n_node=tf.constant([6, 0]), n_edge=tf.constant([0, 0]),
                         edges=None, receivers=None, senders=None, globals=None)
    graphs = autoregressive_connect_graph_dynamic(graphs, exclude_self_edges=False)
    import networkx as nx
    G = nx.MultiDiGraph()
    for sender, receiver in zip(graphs.senders.numpy(), graphs.receivers.numpy()):
        G.add_edge(sender, receiver)
    nx.drawing.draw_circular(G, with_labels=True, node_color=(0, 0, 0), font_color=(1, 1, 1), font_size=25,
                             node_size=1000, arrowsize=30, )
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
        _, senders_array, receivers_array, n_edge_array = tf.while_loop(loop_condition, body, initial_loop_vars)

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

        return graph.replace(senders=tf.stop_gradient(senders),
                             receivers=tf.stop_gradient(receivers),
                             n_edge=tf.stop_gradient(n_edge))


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


class TransformerLayer(AbstractModule):
    """
    Core network which can be used in the EncodeProcessDecode network. Consists of a (full) graph network block
    and a self attention block.
    """

    def __init__(self,
                 num_heads,
                 name=None):
        super(TransformerLayer, self).__init__(name=name)
        self.num_heads = num_heads
        self.ln1 = snt.LayerNorm(axis=-1, eps=1e-6, create_scale=True, create_offset=True, name='layer_norm1')
        self.ln2 = snt.LayerNorm(axis=-1, eps=1e-6, create_scale=True, create_offset=True, name='layer_norm2')
        self.self_attention = SelfAttention()

    @once.once
    def initialize(self, graphs):
        input_node_size = get_shape(graphs.nodes)[-1]
        self.input_node_size = input_node_size

        self.v_linear = MultiHeadLinear(output_size=input_node_size, num_heads=self.num_heads, name='mhl1')  # values
        self.k_linear = MultiHeadLinear(output_size=input_node_size, num_heads=self.num_heads, name='mhl2')  # keys
        self.q_linear = MultiHeadLinear(output_size=input_node_size, num_heads=self.num_heads, name='mhl3')  # queries

        self.FFN = snt.nets.MLP([input_node_size, input_node_size], activate_final=False,
                                name='ffn')  # Feed forward network
        self.output_linear = snt.Linear(output_size=input_node_size, name='output_linear')

    def _build(self, latent):
        self.initialize(latent)
        node_values = self.v_linear(latent.nodes)
        node_keys = self.k_linear(latent.nodes)
        node_queries = self.q_linear(latent.nodes)  # n_node, num_head, F

        # scale queries
        in_degree = tf.math.unsorted_segment_sum(data=tf.ones_like(latent.receivers, dtype=node_queries.dtype),
                                                 segment_ids=latent.receivers,
                                                 num_segments=tf.reduce_sum(latent.n_node))  # n_node
        node_queries /= tf.math.sqrt(in_degree)[:, None, None]  # n_node, F
        attended_latent = self.self_attention(node_values=node_values,
                                              node_keys=node_keys,
                                              node_queries=node_queries,
                                              attention_graph=latent)
        output_nodes = tf.reshape(attended_latent.nodes, (-1, self.num_heads * self.input_node_size))
        output_nodes = self.ln1(self.output_linear(output_nodes) + latent.nodes)
        output_nodes = self.ln2(self.FFN(output_nodes))
        output_graph = latent.replace(nodes=output_nodes)
        return output_graph


class SelfAttentionMessagePassing(AbstractModule):
    """
    Operates on graphs with nodes, and connectivity defined by senders and receivers, and maps to new nodes.
    Optionally uses edges and globals if they are defined.
    """

    def __init__(self, num_heads: int = 1, use_edges=False, use_globals=False, name=None):
        super(SelfAttentionMessagePassing, self).__init__(name=name)
        self.selfattention_core = TransformerLayer(num_heads=num_heads)
        self.layer_norm1 = snt.LayerNorm(1, True, True, name='layer_norm_edges')
        self.layer_norm2 = snt.LayerNorm(1, True, True, name='layer_norm_nodes')
        self.use_globals = use_globals
        self.use_edges = use_edges

    @once.once
    def initialize(self, graphs: GraphsTuple):
        in_node_size = get_shape(graphs.nodes)[-1]
        node_model_fn = lambda: snt.nets.MLP([in_node_size, in_node_size], activate_final=True,
                                             activation=tf.nn.leaky_relu,
                                             name='node_fn')
        edge_model_fn = lambda: snt.nets.MLP([in_node_size, in_node_size],
                                             activate_final=True, activation=tf.nn.leaky_relu,
                                             name='edge_fn')

        self.edge_block = blocks.EdgeBlock(edge_model_fn,
                                           use_edges=self.use_edges,
                                           use_receiver_nodes=True,
                                           use_sender_nodes=True,
                                           use_globals=self.use_globals)

        self.node_block = blocks.NodeBlock(node_model_fn,
                                           use_received_edges=True,
                                           use_sent_edges=True,
                                           use_nodes=True,
                                           use_globals=self.use_globals)

    def _build(self, graphs: GraphsTuple):
        self.initialize(graphs)
        latent_graphs = self.selfattention_core(graphs)
        latent_graphs = self.edge_block(latent_graphs)
        latent_graphs = latent_graphs.replace(edges=self.layer_norm1(latent_graphs.edges))
        latent_graphs = self.node_block(latent_graphs)
        latent_graphs = latent_graphs.replace(nodes=self.layer_norm2(latent_graphs.nodes))
        return latent_graphs


class AutoRegressivePrior(AbstractModule):
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
                 discrete_image_vae: DiscreteImageVAE,
                 discrete_voxel_vae: DiscreteVoxelsVAE,
                 num_heads: int = 1,
                 num_layers: int = 1,
                 temperature: float = 1.,
                 beta: float = 1.,
                 num_token_samples: int = 1,
                 name=None):
        super(AutoRegressivePrior, self).__init__(name=name)
        self.discrete_image_vae = discrete_image_vae
        self.discrete_voxel_vae = discrete_voxel_vae
        assert self.discrete_image_vae.embedding_dim == self.discrete_voxel_vae.embedding_dim, \
            f'Embedding dimensions of prior spaces must be the same, got {self.discrete_image_vae.embedding_dim} and {self.discrete_voxel_vae.embedding_dim}'
        self.starting_node_variable = tf.Variable(
            initial_value=tf.random.truncated_normal((self.discrete_voxel_vae.embedding_dim,)),
            name='starting_token_node')
        self.temperature = temperature
        self.beta = beta
        self.num_token_samples = num_token_samples
        self.embeddings = tf.concat([self.discrete_image_vae.embeddings,
                                     self.discrete_voxel_vae.embeddings],
                                    axis=0)
        self.num_embedding = self.discrete_voxel_vae.num_embedding + self.discrete_image_vae.num_embedding
        self.embedding_dim = self.discrete_image_vae.embedding_dim
        self.num_output = (self.discrete_voxel_vae.voxels_per_dimension // 8) ** 3  # 3 maxpool layers cubed

        message_passing_layers = [SelfAttentionMessagePassing(num_heads=num_heads, name=f"self_attn_mp_{i}")
             for i in range(num_layers)]
        message_passing_layers.append(blocks.NodeBlock(lambda : snt.Linear(self.num_embedding, name='output_linear'),
                                                       use_received_edges=False, use_nodes=True,
                                                       use_globals=False, use_sent_edges=False,
                                                       name='project_block'))
        self.selfattention_core = snt.Sequential(message_passing_layers, name='selfattention_core')

    def _incrementally_decode(self, latent_tokens_2d):
        """
        Args:
            latent_tokens_2d: [batch, H2, W2, embedding_dim]

        Returns:
            GraphsTuple
        """
        batch, H2, W2, _ = get_shape(latent_tokens_2d)
        H3 = W3 = D3 = self.discrete_voxel_vae.voxels_per_dimension // 8
        latent_tokens_3d = tf.zeros((batch, H3, W3, D3, self.embedding_dim))
        concat_graphs = self.construct_concat_graph(latent_tokens_2d, latent_tokens_3d)

        def _core(output_token_idx, latent_graphs):
            """

            Args:
                output_token_idx: which element is being replaced
                latent_graphs: GraphsTuple, nodes are [n_graphs * (H2*W2 + H3*W3*D3), embedding_size]

            Returns:

            """
            batched_latent_graphs = graph_batch_reshape(latent_graphs)

            latent_graphs = self._core(latent_graphs)
            latent_graphs = graph_batch_reshape(latent_graphs)
            latent_logits = latent_graphs.nodes  # batch, H2*W2 + H3 * W3*D3, num_embedding
            latent_logits /= tf.math.reduce_std(latent_logits, axis=-1, keepdims=True)
            latent_logits -= tf.reduce_logsumexp(latent_logits, axis=-1, keepdims=True)

            # [batch, H2*W2 + H3*W3*D3, embedding_dim]
            # [batch, H2*W2 + H3*W3*D3, num_embedding2 + num_embedding3]
            latent_tokens, latent_tokens_onehot = self.sample_latent(latent_logits, S=1)

            _mask = tf.range(H2 * W2 + H3 * W3 * D3) == output_token_idx  # [n_node_per_graph]

            latent_nodes = tf.where(_mask[None, :, None],
                                    latent_tokens,
                                    batched_latent_graphs.nodes)
            batched_latent_graphs = batched_latent_graphs.replace(nodes=latent_nodes)
            latent_graphs = graph_unbatch_reshape(batched_latent_graphs)

            return (output_token_idx + 1, latent_graphs)

        _, latent_graphs = tf.while_loop(
            cond=lambda output_token_idx, state: output_token_idx < (H2 * W2 + H3 * W3 * D3),
            body=lambda output_token_idx, state: _core(output_token_idx, state),
            loop_vars=(tf.convert_to_tensor(H2 * W2), concat_graphs))

        return latent_graphs

    def _build(self, graphs, images):
        """
        Adds another set of nodes to each graph. Autoregressively links all nodes in a graph.

        Args:
            images: [batch, W,H,C]
            graphs: GraphsTuple in standard form.
        Returns:
        """
        # sample 2d embedding
        # [num_samples, batch, H2, W2, num_embedding2], [batch, H2, W2, num_embedding2], [num_samples, batch, H2, W2, embedding_dim]
        token_samples_onehot_2d, latent_logits_2d, latent_tokens_2d = self.sample_2d_embedding(images)
        # print(get_shape(latent_logits_2d), get_shape(latent_tokens_2d))

        # sample 3d embedding
        # [num_samples, batch, H3, W3, D3, num_embedding3], [batch, H3, W3, D3, num_embedding3], [num_samples, batch, H3, W3, D3, embedding_dim]
        token_samples_onehot_3d, latent_logits_3d, latent_tokens_3d = self.sample_3d_embedding(graphs)
        # print(get_shape(latent_logits_3d), get_shape(latent_tokens_3d))

        # auto-regressively model each draw from posterior
        # [num_samples, batch, H2*W2 + H3*W3*D3, num_embedding2 + num_embedding3]
        latent_logits = self.compute_logits(latent_tokens_2d, latent_tokens_3d)

        # # [batch, H2*W2 + H3*W3*D3, embedding_dim], [batch, H2*W2 + H3*W3*D3, num_embedding2 + num_embedding3]
        # latent_tokens, latent_tokens_onehot = self.sample_latent(latent_logits)

        # get likelihoods
        _, batch, H2, W2, _ = get_shape(latent_tokens_2d)
        _, batch, H3, W3, D3, _ = get_shape(latent_tokens_3d)


        # predict_latent_tokens_2d = tf.reshape(latent_tokens[:, :, :H2 * W2, :], (
        #     self.num_token_samples, batch, H2, W2, self.embedding_dim))  # num_samples, batch, H2,W2, embedding_dim
        # predict_latent_tokens_onehot_2d = tf.reshape(latent_tokens_onehot[:, :, :H2 * W2, :], (
        #     self.num_token_samples, batch, H2, W2, self.num_embedding))  # num_samples, batch, H2,W2, num_embedding
        # predict_latent_logits_2d = tf.reshape(latent_logits[:, :, :H2 * W2, :], (
        #     self.num_token_samples, batch, H2, W2, self.num_embedding))  # num_samples, batch, H2,W2, num_embedding
        #
        # predict_latent_tokens_3d = tf.reshape(latent_tokens[:, :, H2 * W2:, :], (
        #     self.num_token_samples, batch, H3, W3, D3,
        #     self.embedding_dim))  # num_samples, batch, H3 , W3,D3, embedding_dim
        # predict_latent_tokens_onehot_3d = tf.reshape(latent_tokens_onehot[:, :, H2 * W2:, :], (
        #     self.num_token_samples, batch, H3, W3, D3,
        #     self.num_embedding))  # num_samples, batch, H3 , W3,D3, num_embedding
        # predict_latent_logits_3d = tf.reshape(latent_logits[:, :, H2 * W2:, :], (
        #     self.num_token_samples, batch, H3, W3, D3,
        #     self.num_embedding))  # num_samples, batch, H3 , W3,D3, num_embedding

        logb_2d, logb_3d, mu_2d, mu_3d = self.compute_log_likelihood_params(latent_tokens_2d, latent_tokens_3d)

        log_likelihood = self.log_likelihood(graphs, images, logb_2d, logb_3d, mu_2d, mu_3d)

        # kl-terms
        kl_term, log_prob_prior = self.compute_kl_term(latent_logits, latent_logits_2d, latent_logits_3d,
                                               token_samples_onehot_2d, token_samples_onehot_3d)

        var_exp = tf.reduce_mean(log_likelihood, axis=0)  # [batch]
        kl_div = tf.reduce_mean(kl_term, axis=0)  # [batch]
        elbo = var_exp - self.beta * kl_div  # batch
        loss = - tf.reduce_mean(elbo)  # scalar

        entropy = -tf.reduce_sum(tf.math.exp(log_prob_prior) * log_prob_prior, axis=-1)  # #num_samples, batch
        perplexity = 2. ** (entropy / tf.math.log(2.))  # [S, batch]
        mean_perplexity = tf.reduce_mean(perplexity)  # scalar

        return dict(loss=loss,
                    metrics=dict(var_exp=var_exp,
                                 kl_div=kl_div,
                                 mean_perplexity=mean_perplexity))

    def compute_kl_term(self, latent_logits, latent_logits_2d, latent_logits_3d, token_samples_onehot_2d,
                token_samples_onehot_3d):
        _, batch, H2, W2, _ = get_shape(token_samples_onehot_2d)
        _, batch, H3, W3, D3, _ = get_shape(token_samples_onehot_3d)
        log_prob_q_2d = self.discrete_image_vae.log_prob_q(latent_logits_2d,
                                                           token_samples_onehot_2d)  # num_samples, batch, H, W
        log_prob_q_2d *= (H3 * W3 * D3)
        log_prob_q_3d = self.discrete_voxel_vae.log_prob_q(latent_logits_3d,
                                                           token_samples_onehot_3d)  # num_samples, batch, H, W, D
        log_prob_q_3d *= (H2 * W2)
        prior_dist = tfp.distributions.RelaxedOneHotCategorical(self.temperature, logits=latent_logits)
        # num_samples, batch, H2, W2, num_embedding2 + num_embedding3
        token_samples_onehot_2d = tf.concat([token_samples_onehot_2d,
                                             tf.zeros((self.num_token_samples, batch, H2, W2,
                                                       self.discrete_voxel_vae.num_embedding))
                                             ],
                                            axis=-1)
        # num_samples, batch, H2*W2, num_embedding2 + num_embedding3
        token_samples_onehot_2d = tf.reshape(token_samples_onehot_2d,
                                             (self.num_token_samples, batch, H2 * W2, self.num_embedding))
        # num_samples, batch, H3, W3, D3 num_embedding2 + num_embedding3
        token_samples_onehot_3d = tf.concat([tf.zeros((self.num_token_samples, batch, H3, W3, D3,
                                                       self.discrete_image_vae.num_embedding)),
                                             token_samples_onehot_3d
                                             ],
                                            axis=-1)
        # num_samples, batch, H3*W3*D3, num_embedding2 + num_embedding3
        token_samples_onehot_3d = tf.reshape(token_samples_onehot_3d,
                                             (self.num_token_samples, batch, H3 * W3 * D3, self.num_embedding))
        # num_samples, batch, H2*W2 + H3 * W3*D3, num_embedding2 + num_embedding3
        latent_tokens_onehot = tf.concat([token_samples_onehot_2d, token_samples_onehot_3d], axis=-2)
        log_prob_prior = prior_dist.log_prob(latent_tokens_onehot)  # num_samples, batch, H2*W2 + H3 * W3*D3
        kl_term = tf.reduce_sum(log_prob_q_2d, axis=[-1, -2]) + tf.reduce_sum(log_prob_q_3d, axis=[-1, -2, -3]) \
                  - tf.reduce_sum(log_prob_prior, axis=-1)  # num_samples, batch
        return kl_term, log_prob_prior

    def log_likelihood(self, graphs, images, logb_2d, logb_3d, mu_2d, mu_3d):
        """
        Args:
            graphs: batch graphs in a GraphTuples in standard form.
            images: [batch, H', W', C]
            logb_2d: [num_samples, batch, H', W', C]
            logb_3d: [num_samples, batch, H', W', D', C]
            mu_2d: [num_samples, batch, H', W', C]
            mu_3d: [num_samples, batch, H', W', D', C]

        Returns:
            [num_samples, batch]
        """
        log_likelihood_2d = self.discrete_image_vae.log_likelihood(images, mu_2d, logb_2d)  # [num_samples, batch]
        log_likelihood_3d = self.discrete_voxel_vae.log_likelihood(graphs, mu_3d, logb_3d)  # [num_samples, batch]
        log_likelihood = log_likelihood_2d + log_likelihood_3d  # [num_samples, batch]
        return log_likelihood

    def compute_log_likelihood_params(self, latent_tokens_2d, latent_tokens_3d):
        """
        Args:
            predict_latent_tokens_2d: [num_samples, batch, H2, W2, embedding_dim]
            predict_latent_tokens_3d: [num_samples, batch, H3, W3, D3, embedding_dim]

        Returns:
            logb_2d, logb_3d, mu_2d, mu_3d
        """
        mu_2d, logb_2d = self.discrete_image_vae.compute_likelihood_parameters(
            latent_tokens_2d)  # [num_samples, batch, H', W', C], [num_samples, batch, H', W', C]
        mu_3d, logb_3d = self.discrete_voxel_vae.compute_likelihood_parameters(
            latent_tokens_3d)  # [num_samples, batch, H', W', D', C], [num_samples, batch, H', W', D', C]
        return logb_2d, logb_3d, mu_2d, mu_3d

    def sample_latent(self, latent_logits, S=None):
        """
        Args:
            latent_logits: [batch, H2*W2 + H3*W3*D3, num_embedding2 + num_embedding3]
            S = num_samples or None
        Returns:
            latent_tokens: [batch, H2*W2 + H3*W3*D3, embedding_dim]
            latent_tokens_onehot: [batch, H2*W2 + H3*W3*D3, num_embedding2 + num_embedding3]
        """
        if S is None:
            S = self.num_token_samples
        prior_dist = tfp.distributions.RelaxedOneHotCategorical(self.temperature, logits=latent_logits)
        latent_tokens_onehot = prior_dist.sample(S)  # num_samples, batch, H2*W2 + H3 * W3*D3, num_embedding
        latent_tokens = tf.einsum("sbne,ed->sbnd", latent_tokens_onehot, self.embeddings)
        return latent_tokens, latent_tokens_onehot

    def compute_logits(self, latent_tokens_2d, latent_tokens_3d):
        """

        Args:
            latent_tokens_2d: [num_samples, batch, H2, W2, embedding_dim]
            latent_tokens_3d: [num_samples, batch, H3, W3, D3, embedding_dim]

        Returns:
            latent_logits: [num_samples, batch, H2*W2 + H3*W3*D3, num_embedding2 + num_embedding3]
        """
        _, batch, H2, W2, _ = get_shape(latent_tokens_2d)
        latent_tokens_2d = tf.reshape(latent_tokens_2d, [self.num_token_samples*batch, H2, W2, self.embedding_dim])
        _, batch, H3, W3, D3, _ = get_shape(latent_tokens_3d)
        latent_tokens_3d = tf.reshape(latent_tokens_3d, [self.num_token_samples * batch, H3, W3, D3, self.embedding_dim])
        if latent_tokens_3d is not None:
            latent_graphs = self.forced_teacher_model(latent_tokens_2d, latent_tokens_3d)
        else:
            latent_graphs = self._incrementally_decode(latent_tokens_2d)
        latent_graphs = graph_batch_reshape(latent_graphs)
        latent_logits = latent_graphs.nodes  #num_samples*batch, H2*W2 + H3 * W3*D3, num_embedding
        latent_logits = tf.reshape(latent_logits, (self.num_token_samples, batch, H2*W2 + H3 * W3*D3, self.num_embedding))#num_samples, batch, H2*W2 + H3 * W3*D3, num_embedding
        latent_logits /= tf.math.reduce_std(latent_logits, axis=-1, keepdims=True)
        latent_logits -= tf.reduce_logsumexp(latent_logits, axis=-1, keepdims=True)
        return latent_logits

    def forced_teacher_model(self, latent_tokens_2d, latent_tokens_3d):
        """
        Args:
            latent_tokens_2d: [num_samples*batch, H2, W2, embedding_dim]
            latent_tokens_3d: [num_samples*batch, H3, W3, D3, embedding_dim]

        Returns:
            num_samples*batch graphs in a GraphsTuple in standard form.
        """
        concat_graphs = self.construct_concat_graph(latent_tokens_2d, latent_tokens_3d)
        latent_graphs = self._core(concat_graphs)
        return latent_graphs

    def _core(self, concat_graphs):
        return self.selfattention_core(concat_graphs)

    def construct_concat_graph(self, latent_tokens_2d, latent_tokens_3d):
        """
        Args:
            latent_tokens_2d: [batch, H2, W2, num_embeddings]
            latent_tokens_3d: [batch, H3, W3, D3, num_embeddings]

        Returns:
            concat_graphs: GraphsTuple, nodes are [n_graphs * (num_input + num_output), embedding_size]
        """
        batch, H2, W2, _ = get_shape(latent_tokens_2d)
        batch, H3, W3, D3, _ = get_shape(latent_tokens_3d)
        flat_latent_tokens_2d = tf.reshape(latent_tokens_2d, (batch, H2 * W2, self.discrete_image_vae.embedding_dim))
        flat_latent_tokens_3d = tf.reshape(latent_tokens_3d,
                                           (batch, H3 * W3 * D3, self.discrete_voxel_vae.embedding_dim))
        nodes = tf.concat([flat_latent_tokens_2d,
                           tf.tile(self.starting_node_variable[None, None, :], [batch, 1, 1]),
                           flat_latent_tokens_3d[:, :-1, :]], axis=1)
        n_node = tf.fill([batch], H2 * W2 + H3 * W3 * D3)
        n_edge = tf.zeros_like(n_node)
        data_dict = dict(nodes=nodes, edges=None, senders=None, receivers=None, globals=None,
                         n_node=n_node, n_edge=n_edge)
        concat_graphs = GraphsTuple(**data_dict)
        concat_graphs = graph_unbatch_reshape(concat_graphs)  # [n_graphs * (num_input + num_output), embedding_size]
        # nodes, senders, receivers, globals
        concat_graphs = autoregressive_connect_graph_dynamic(concat_graphs)
        return concat_graphs

    def sample_3d_embedding(self, graphs):
        """
        Args:
            graphs:

        Returns:
            token_samples_onehot_3d: [num_samples, batch, H, W, D, num_embeddings]
            latent_logits_3d: [batch, H, W, D, num_embeddings]
            latent_tokens_3d: [num_samples, batch, H, W, D, embedding_dim]
        """
        latent_logits_3d = self.discrete_voxel_vae.compute_logits(graphs)  # [batch, H, W, D, num_embeddings]
        token_samples_onehot_3d, latent_tokens_3d = self.discrete_voxel_vae.sample_latent(latent_logits_3d,
                                                                                          self.discrete_voxel_vae.temperature,
                                                                                          self.num_token_samples)  # [num_samples, batch, H, W, D, num_embeddings], [num_samples, batch, H, W, D, embedding_size]
        return token_samples_onehot_3d, latent_logits_3d, latent_tokens_3d

    def sample_2d_embedding(self, images):
        """
        Args:
            images:

        Returns:
            token_samples_onehot_2d: [num_samples, batch, H, W, num_embeddings]
            latent_logits_2d: [batch, H, W, num_embeddings]
            latent_tokens_2d: [num_samples, batch, H, W, embedding_size]
        """
        latent_logits_2d = self.discrete_image_vae.compute_logits(images)  # [batch, H, W, num_embeddings]
        token_samples_onehot_2d, latent_tokens_2d = self.discrete_image_vae.sample_latent(latent_logits_2d,
                                                                                          self.discrete_image_vae.temperature,
                                                                                          self.num_token_samples)  # [num_samples, batch, H, W, num_embeddings], [1, batch, H, W, embedding_size]
        return token_samples_onehot_2d, latent_logits_2d, latent_tokens_2d

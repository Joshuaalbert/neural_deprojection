from graph_nets import blocks
import tensorflow as tf
import sonnet as snt
from graph_nets.graphs import GraphsTuple
from graph_nets import utils_tf
from neural_deprojection.graph_net_utils import AbstractModule, get_shape, graph_batch_reshape, graph_unbatch_reshape, \
    grid_graphs
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
    """Adds edges to a graph by auto-regressively the nodes.

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

        _, _, d_k = get_shape(node_keys)
        node_queries /= tf.math.sqrt(tf.cast(d_k, node_queries.dtype))  # n_node, F
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
                                             activation=tf.nn.relu,
                                             name='node_fn')
        edge_model_fn = lambda: snt.nets.MLP([in_node_size, in_node_size],
                                             activate_final=True, activation=tf.nn.relu,
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
    This models an auto-regressive joint distribution, which is typically for modelling a prior.



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
                 compute_temperature: callable = None,
                 embedding_dim: int = 16,
                 beta: float = 1.,
                 num_token_samples: int = 1,
                 name=None):
        super(AutoRegressivePrior, self).__init__(name=name)
        self.discrete_image_vae = discrete_image_vae
        self.discrete_voxel_vae = discrete_voxel_vae
        self.embedding_dim = embedding_dim

        self.compute_temperature = compute_temperature
        self.beta = tf.convert_to_tensor(beta, dtype=tf.float32)
        self.num_token_samples = num_token_samples

        self.num_embedding = self.discrete_voxel_vae.num_embedding + self.discrete_image_vae.num_embedding

        self.embeddings = tf.Variable(
            initial_value=tf.random.truncated_normal((self.num_embedding, self.embedding_dim)),
            name='embeddings')

        self.num_output = (self.discrete_voxel_vae.voxels_per_dimension // 8) ** 3  # 3 maxpool layers cubed

        message_passing_layers = [SelfAttentionMessagePassing(num_heads=num_heads, name=f"self_attn_mp_{i}")
                                  for i in range(num_layers)]
        message_passing_layers.append(blocks.NodeBlock(lambda: snt.Linear(self.num_embedding, name='output_linear'),
                                                       use_received_edges=False, use_nodes=True,
                                                       use_globals=False, use_sent_edges=False,
                                                       name='project_block'))
        self.selfattention_core = snt.Sequential(message_passing_layers, name='selfattention_core')

    @property
    def temperature(self):
        return self.compute_temperature(self.epoch)

    @once.once
    def initialize_positional_encodings(self, nodes):
        _, n_node, _ = get_shape(nodes)
        self.positional_encodings = tf.Variable(
            initial_value=tf.random.truncated_normal((n_node, self.embedding_dim)),
            name='positional_encodings')

    @tf.function(input_signature=[tf.TensorSpec((None, None, None, None), tf.float32)])
    def deproject_images(self, images):
        """
        For a batch of images samples a 3D medium consistent with the image.

        Args:
            images: [batch, H2, W2, C2]

        Returns:
            mu, b: mean, uncertainty of images [batch, H3, W2, D3, C3]
        """
        logits_2d = self.discrete_image_vae.compute_logits(images)#[batch, W2, H2, num_embeddings2]
        dist_2d = tfp.distributions.OneHotCategorical(logits=logits_2d, dtype=images.dtype)
        token_samples_onehot_2d = dist_2d.sample(1)#[1, batch, W2, H2, num_embeddings2]
        token_samples_onehot_2d = token_samples_onehot_2d[0]#batch, W2, H2, num_embeddings2
        token_samples_onehot_3d = self._incrementally_decode(token_samples_onehot_2d)#batch, H3,W3,D3, num_embedding3
        latent_token_samples_3d = tf.einsum("bwhdn,nm->bwhdm", token_samples_onehot_3d,
                                  self.discrete_voxel_vae.embeddings)  # [batch, W3, H3, D3, embedding_size3]
        mu_3d, logb_3d = self.discrete_voxel_vae.compute_likelihood_parameters(latent_token_samples_3d[None])# [1, batch, H', W', D' C], [1, batch, H', W', D' C]_
        mu_3d = mu_3d[0]
        logb_3d = logb_3d[0]
        b_3d = tf.math.exp(logb_3d)
        return mu_3d, b_3d


    def _incrementally_decode(self, token_samples_onehot_2d):
        """
        Args:
            token_samples_onehot_2d: [batch, H2, W2, num_embedding2]

        Returns:
            token_samples_onehot_3d: [batch, H3,W3,D3, num_embedding3]
        """

        batch, H2, W2, _ = get_shape(token_samples_onehot_2d)
        H3 = W3 = D3 = self.discrete_voxel_vae.voxels_per_dimension // self.discrete_voxel_vae.shrink_factor

        token_samples_onehot_2d = tf.reshape(token_samples_onehot_2d,
                                             (batch, H2 * W2, self.discrete_image_vae.num_embedding))

        N, _ = get_shape(self.positional_encodings)

        # num_samples*batch, H2*W2, num_embedding2
        latent_tokens_2d = tf.einsum("bne,ed->bnd", token_samples_onehot_2d,
                                     self.embeddings[:self.discrete_image_vae.num_embedding, :])
        #batch, H3*W3*D3, num_embedding3
        output_token_samples_onehot_3d = tf.zeros((batch, H3*W3*D3, self.discrete_voxel_vae.num_embedding))


        latent_tokens_3d = tf.zeros((batch, H3*W3*D3, self.embedding_dim))
        concat_graphs = self.construct_concat_graph(latent_tokens_2d, latent_tokens_3d)
        concat_graphs_data_dict = {key: value for key, value in concat_graphs._asdict().items() if value is not None}

        def _core(output_token_idx, latent_graphs_data_dict, output_token_samples_onehot_3d):
            """

            Args:
                output_token_idx: which element is being replaced
                latent_graphs: GraphsTuple, nodes are [n_graphs * (H2*W2 + H3*W3*D3), embedding_size]
                output_token_samples_onehot_3d: [batch, H3*W3*D3, num_embedding3]

            Returns:

            """

            latent_graphs = GraphsTuple(**latent_graphs_data_dict, edges=None, globals=None)
            batched_latent_graphs = graph_batch_reshape(latent_graphs)

            latent_graphs = self._core(latent_graphs)
            latent_graphs = graph_batch_reshape(latent_graphs)
            latent_logits = latent_graphs.nodes  # batch, H2*W2 + H3 * W3*D3, num_embedding
            latent_logits /= tf.math.reduce_std(latent_logits, axis=-1, keepdims=True)
            latent_logits -= tf.reduce_logsumexp(latent_logits, axis=-1, keepdims=True)

            # [batch, H2*W2 + H3*W3*D3, embedding_dim]
            # [batch, H2*W2 + H3*W3*D3, num_embedding2 + num_embedding3]
            prior_dist = tfp.distributions.OneHotCategorical(logits=latent_logits, dtype=latent_logits.dtype)
            prior_token_samples_onehot = prior_dist.sample(1)# 1, batch, H2*W2 + H3 * W3*D3, num_embedding
            prior_token_samples_onehot = prior_token_samples_onehot[0]#batch, H2*W2 + H3 * W3*D3, num_embedding
            prior_token_samples_onehot_3d = prior_token_samples_onehot[:, H2*W2:, self.discrete_image_vae.num_embedding:]#batch, H3*W3*D3, num_embedding3

            _mask = tf.range(H3 * W3 * D3) == output_token_idx  # [H3*W3*D3]

            output_token_samples_onehot_3d = tf.where(_mask[None, :, None],
                                                      prior_token_samples_onehot_3d,
                                                      output_token_samples_onehot_3d
                                                      )

            prior_token_samples = tf.einsum("bne,ed->bnd",prior_token_samples_onehot, self.embeddings)#batch, H2*W2 + H3 * W3*D3, embedding_dim
            prior_token_samples += self.positional_encodings

            _mask = tf.range(H2*W2 + H3 * W3 * D3) == H2*W2 + output_token_idx  # [H2*W2 + H3*W3*D3]

            latent_nodes = tf.where(_mask[None, :, None],
                                    prior_token_samples,
                                    batched_latent_graphs.nodes)
            batched_latent_graphs = batched_latent_graphs.replace(nodes=latent_nodes)
            latent_graphs = graph_unbatch_reshape(batched_latent_graphs)
            latent_graphs_data_dict = {key: value for key, value in latent_graphs._asdict().items() if
                                       value is not None}
            return (output_token_idx + 1, latent_graphs_data_dict, output_token_samples_onehot_3d)



        _, latent_graphs_data_dict, output_token_samples_onehot_3d = tf.while_loop(
            cond=lambda output_token_idx, _1, _2: output_token_idx < (H3 * W3 * D3),
            body=_core,
            loop_vars=(tf.convert_to_tensor(0), concat_graphs_data_dict, output_token_samples_onehot_3d))

        # latent_graphs = GraphsTuple(**latent_graphs_data_dict, edges=None, globals=None)

        output_token_samples_onehot_3d = tf.reshape(output_token_samples_onehot_3d,
                                                    (batch, H3,W3,D3, self.discrete_voxel_vae.num_embedding))
        return output_token_samples_onehot_3d

    def write_summary(self, images, graphs,
                      latent_logits_2d, latent_logits_3d,
                      token_samples_onehot_2d, token_samples_onehot_3d,
                      latent_tokens_2d, latent_tokens_3d,
                      prior_latent_logits_2d, prior_latent_logits_3d,
                      kl_term_2d, kl_term_3d):

        kl_div_2d = tf.reduce_mean(kl_term_2d, axis=0)  # [batch]
        kl_div_3d = tf.reduce_mean(kl_term_3d, axis=0)  # [batch]

        mu_2d, logb_2d = self.discrete_image_vae.compute_likelihood_parameters(
            latent_tokens_2d)  # [num_samples, batch, H', W', C], [num_samples, batch, H', W', C]
        log_likelihood_2d = self.discrete_image_vae.log_likelihood(images, mu_2d, logb_2d)  # [num_samples, batch]

        var_exp_2d = tf.reduce_mean(log_likelihood_2d, axis=0)  # [batch]

        mu_3d, logb_3d = self.discrete_voxel_vae.compute_likelihood_parameters(
            latent_tokens_3d)  # [num_samples, batch, H', W', D', C], [num_samples, batch, H', W', D', C]
        log_likelihood_3d = self.discrete_voxel_vae.log_likelihood(graphs, mu_3d, logb_3d)  # [num_samples, batch]

        var_exp_3d = tf.reduce_mean(log_likelihood_3d, axis=0)  # [batch]

        var_exp = tf.reduce_mean(log_likelihood_2d + log_likelihood_3d, axis=0)

        entropy_2d = -tf.reduce_mean(tf.math.exp(prior_latent_logits_2d) * prior_latent_logits_2d, axis=[-1])  #
        perplexity_2d = 2. ** (entropy_2d/tf.math.log(2.))  #
        mean_perplexity_2d = tf.reduce_mean(perplexity_2d)  # scalar

        entropy_3d = -tf.reduce_mean(tf.math.exp(prior_latent_logits_3d) * prior_latent_logits_3d, axis=[-1])  #
        perplexity_3d = 2. ** (entropy_3d / tf.math.log(2.))  #
        mean_perplexity_3d = tf.reduce_mean(perplexity_3d)  # scalar

        log_token_samples_onehot_2d_prior, token_samples_onehot_2d_prior, latent_tokens_2d_prior = self.discrete_image_vae.sample_latent(
            prior_latent_logits_2d[0],
            self.temperature,
            self.num_token_samples)  # [num_samples, batch, H, W, num_embeddings], [num_samples, batch, H, W, embedding_size]
        mu_2d_prior, logb_2d_prior = self.discrete_image_vae.compute_likelihood_parameters(
            latent_tokens_2d_prior)  # [num_samples, batch, H', W', C], [num_samples, batch, H', W', C]

        log_token_samples_onehot_3d_prior, token_samples_onehot_3d_prior, latent_tokens_3d_prior = self.discrete_voxel_vae.sample_latent(
            prior_latent_logits_3d[0],
            self.temperature,
            self.num_token_samples)  # [num_samples, batch, H, W, num_embeddings], [num_samples, batch, H, W, embedding_size]
        mu_3d_prior, logb_3d_prior = self.discrete_voxel_vae.compute_likelihood_parameters(
            latent_tokens_3d_prior)  # [num_samples, batch, H', W', C], [num_samples, batch, H', W', C]

        tf.summary.scalar('perplexity_2d_prior', mean_perplexity_2d, step=self.step)
        tf.summary.scalar('perplexity_3d_prior', mean_perplexity_3d, step=self.step)
        tf.summary.scalar('var_exp_3d', tf.reduce_mean(var_exp_3d), step=self.step)
        tf.summary.scalar('var_exp_2d', tf.reduce_mean(var_exp_2d), step=self.step)
        tf.summary.scalar('kl_div_2d', tf.reduce_mean(kl_div_2d), step=self.step)
        tf.summary.scalar('kl_div_3d', tf.reduce_mean(kl_div_3d), step=self.step)
        tf.summary.scalar('var_exp', tf.reduce_mean(var_exp), step=self.step)
        tf.summary.scalar('kl_div', tf.reduce_mean(kl_div_2d + kl_div_3d), step=self.step)
        tf.summary.scalar('temperature_2d', self.discrete_image_vae.temperature, step=self.step)
        tf.summary.scalar('temperature_3d', self.discrete_voxel_vae.temperature, step=self.step)
        tf.summary.scalar('temperature', self.temperature, step=self.step)
        tf.summary.scalar('beta', self.beta, step=self.step)

        projected_mu = tf.reduce_sum(mu_3d[0], axis=-2)  # [batch, H', W', C]
        projected_mu_prior = tf.reduce_sum(mu_3d_prior[0], axis=-2)  # [batch, H', W', C]
        voxels = grid_graphs(graphs, self.discrete_voxel_vae.voxels_per_dimension)  # [batch, H', W', D', C]
        projected_img = tf.reduce_sum(voxels, axis=-2)  # [batch, H', W', C]
        for i in range(self.discrete_voxel_vae.num_channels):
            vmin = tf.reduce_min(projected_mu[..., i])
            vmax = tf.reduce_max(projected_mu[..., i])
            _projected_mu = (projected_mu[..., i:i + 1] - vmin) / (vmax - vmin)  # batch, H', W', 1
            _projected_mu = tf.clip_by_value(_projected_mu, 0., 1.)

            vmin = tf.reduce_min(projected_mu_prior[..., i])
            vmax = tf.reduce_max(projected_mu_prior[..., i])
            _projected_mu_prior = (projected_mu_prior[..., i:i + 1] - vmin) / (vmax - vmin)  # batch, H', W', 1
            _projected_mu_prior = tf.clip_by_value(_projected_mu_prior, 0., 1.)

            vmin = tf.reduce_min(projected_img[..., i])
            vmax = tf.reduce_max(projected_img[..., i])
            _projected_img = (projected_img[..., i:i + 1] - vmin) / (vmax - vmin)  # batch, H', W', 1
            _projected_img = tf.clip_by_value(_projected_img, 0., 1.)

            tf.summary.image(f'voxels_predict[{i}]', _projected_mu, step=self.step)
            tf.summary.image(f'voxels_predict_prior[{i}]', _projected_mu_prior, step=self.step)
            tf.summary.image(f'voxels_actual[{i}]', _projected_img, step=self.step)

        batch, H3, W3, D3, _ = get_shape(latent_logits_3d)
        _latent_logits_3d = latent_logits_3d  # [batch, H, W, D, num_embeddings]
        _latent_logits_3d -= tf.reduce_min(_latent_logits_3d, axis=-1, keepdims=True)
        _latent_logits_3d /= tf.reduce_max(_latent_logits_3d, axis=-1, keepdims=True)
        _latent_logits_3d = tf.reshape(_latent_logits_3d,
                                       [batch, H3 * W3 * D3, self.discrete_voxel_vae.num_embedding,
                                        1])  # [batch, H*W*D, num_embedding, 1]
        tf.summary.image('latent_logits_3d', _latent_logits_3d, step=self.step)

        token_sample_onehot_3d = token_samples_onehot_3d[0]  # [batch, H, W, D, num_embeddings]
        token_sample_onehot_3d = tf.reshape(token_sample_onehot_3d,
                                            [batch, H3 * W3 * D3, self.discrete_voxel_vae.num_embedding,
                                             1])  # [batch, H*W*D, num_embedding, 1]
        tf.summary.image('latent_samples_onehot_3d', token_sample_onehot_3d, step=self.step)

        _latent_logits_3d_prior = prior_latent_logits_3d[0]  # [batch, H, W, D, num_embeddings]
        _latent_logits_3d_prior -= tf.reduce_min(_latent_logits_3d_prior, axis=-1, keepdims=True)
        _latent_logits_3d_prior /= tf.reduce_max(_latent_logits_3d_prior, axis=-1, keepdims=True)
        _latent_logits_3d_prior = tf.reshape(_latent_logits_3d_prior,
                                             [batch, H3 * W3 * D3, self.discrete_voxel_vae.num_embedding,
                                              1])  # [batch, H*W*D, num_embedding, 1]
        tf.summary.image('latent_logits_3d_prior', _latent_logits_3d_prior, step=self.step)

        token_sample_onehot_3d_prior = token_samples_onehot_3d_prior[0]  # [batch, H, W, D, num_embeddings]
        token_sample_onehot_3d_prior = tf.reshape(token_sample_onehot_3d_prior,
                                                  [batch, H3 * W3 * D3, self.discrete_voxel_vae.num_embedding,
                                                   1])  # [batch, H*W*D, num_embedding, 1]
        tf.summary.image('latent_samples_onehot_3d_prior', token_sample_onehot_3d_prior, step=self.step)

        _mu = mu_2d[0]  # [batch, H', W', C]
        _mu_prior = mu_2d_prior[0]  # [batch, H', W', C]
        _img = images  # [batch, H', W', C]
        for i in range(self.discrete_image_vae.num_channels):
            vmin = tf.reduce_min(_mu[..., i])
            vmax = tf.reduce_max(_mu[..., i])
            _projected_mu = (_mu[..., i:i + 1] - vmin) / (vmax - vmin)  # batch, H', W', 1
            _projected_mu = tf.clip_by_value(_projected_mu, 0., 1.)

            vmin = tf.reduce_min(_mu_prior[..., i])
            vmax = tf.reduce_max(_mu_prior[..., i])
            _projected_mu_prior = (_mu[..., i:i + 1] - vmin) / (vmax - vmin)  # batch, H', W', 1
            _projected_mu_prior = tf.clip_by_value(_projected_mu_prior, 0., 1.)

            vmin = tf.reduce_min(_img[..., i])
            vmax = tf.reduce_max(_img[..., i])
            _projected_img = (_img[..., i:i + 1] - vmin) / (vmax - vmin)  # batch, H', W', 1
            _projected_img = tf.clip_by_value(_projected_img, 0., 1.)

            tf.summary.image(f'image_predict[{i}]', _projected_mu, step=self.step)
            tf.summary.image(f'image_predict_prior[{i}]', _projected_mu_prior, step=self.step)
            tf.summary.image(f'image_actual[{i}]', _projected_img, step=self.step)

        batch, H2, W2, _ = get_shape(latent_logits_2d)
        _latent_logits_2d = latent_logits_2d  # [batch, H, W, num_embeddings]
        _latent_logits_2d -= tf.reduce_min(_latent_logits_2d, axis=-1, keepdims=True)
        _latent_logits_2d /= tf.reduce_max(_latent_logits_2d, axis=-1, keepdims=True)
        _latent_logits_2d = tf.reshape(_latent_logits_2d,
                                       [batch, H2 * W2, self.discrete_image_vae.num_embedding,
                                        1])  # [batch, H*W, num_embedding, 1]
        tf.summary.image('latent_logits_2d', _latent_logits_2d, step=self.step)

        token_sample_onehot = token_samples_onehot_2d[0]  # [batch, H, W, num_embeddings]
        token_sample_onehot = tf.reshape(token_sample_onehot,
                                         [batch, H2 * W2, self.discrete_image_vae.num_embedding,
                                          1])  # [batch, H*W, num_embedding, 1]
        tf.summary.image('latent_samples_onehot_2d', token_sample_onehot, step=self.step)

        _latent_logits_2d_prior = prior_latent_logits_2d[0]  # [batch, H, W, num_embeddings]
        _latent_logits_2d_prior -= tf.reduce_min(_latent_logits_2d_prior, axis=-1, keepdims=True)
        _latent_logits_2d_prior /= tf.reduce_max(_latent_logits_2d_prior, axis=-1, keepdims=True)
        _latent_logits_2d_prior = tf.reshape(_latent_logits_2d_prior,
                                             [batch, H2 * W2, self.discrete_image_vae.num_embedding,
                                              1])  # [batch, H*W, num_embedding, 1]
        tf.summary.image('latent_logits_2d_prior', _latent_logits_2d_prior, step=self.step)

        token_sample_onehot = token_samples_onehot_2d_prior[0]  # [batch, H, W, num_embeddings]
        token_sample_onehot = tf.reshape(token_sample_onehot,
                                         [batch, H2 * W2, self.discrete_image_vae.num_embedding,
                                          1])  # [batch, H*W, num_embedding, 1]
        tf.summary.image('latent_samples_onehot_2d_prior', token_sample_onehot, step=self.step)

    def _build(self, graphs, images):

        latent_logits_2d = self.discrete_image_vae.compute_logits(images)  # [batch, H, W, num_embeddings]
        log_token_samples_onehot_2d, token_samples_onehot_2d, latent_tokens_2d = self.discrete_image_vae.sample_latent(
            latent_logits_2d,
            self.discrete_image_vae.temperature,
            self.num_token_samples)  # [num_samples, batch, H, W, num_embeddings], [num_samples, batch, H, W, embedding_size]

        latent_logits_3d = self.discrete_voxel_vae.compute_logits(graphs)  # [batch, H, W, D, num_embeddings]
        log_token_samples_onehot_3d, token_samples_onehot_3d, latent_tokens_3d = self.discrete_voxel_vae.sample_latent(
            latent_logits_3d,
            self.discrete_voxel_vae.temperature,
            self.num_token_samples)  # [num_samples, batch, H, W, D, num_embeddings], [num_samples, batch, H, W, D, embedding_size]

        prior_latent_logits_2d, prior_latent_logits_3d = self.compute_prior_logits(token_samples_onehot_2d,
                                                                                   token_samples_onehot_3d)

        kl_term_2d, kl_term_3d, log_prob_prior_2d, log_prob_prior_3d = self.compute_kl_term(latent_logits_2d,
                                                                                            latent_logits_3d,
                                                                                            log_token_samples_onehot_2d,
                                                                                            log_token_samples_onehot_3d,
                                                                                            prior_latent_logits_2d,
                                                                                            prior_latent_logits_3d)

        kl_term = kl_term_2d + kl_term_3d  # [num_samples, batch]

        kl_div = tf.reduce_mean(kl_term, axis=0)  # [batch]
        # elbo = tf.stop_gradient(var_exp) - self.beta * kl_div  # batch
        elbo = - kl_div  # batch

        loss = - tf.reduce_mean(elbo)  # scalar

        if self.step % 10 == 0:
            self.write_summary(images, graphs,
                      latent_logits_2d, latent_logits_3d,
                      token_samples_onehot_2d, token_samples_onehot_3d,
                      latent_tokens_2d, latent_tokens_3d,
                      prior_latent_logits_2d, prior_latent_logits_3d,
                      kl_term_2d, kl_term_3d)

        return dict(loss=loss)

    def compute_kl_term(self, latent_logits_2d, latent_logits_3d, log_token_samples_onehot_2d,
                        log_token_samples_onehot_3d, prior_latent_logits_2d, prior_latent_logits_3d):
        q_dist_2d = tfp.distributions.ExpRelaxedOneHotCategorical(self.discrete_image_vae.temperature,
                                                                  logits=latent_logits_2d)
        log_prob_q_2d = q_dist_2d.log_prob(log_token_samples_onehot_2d)  # num_samples, batch, H2, W2
        q_dist_3d = tfp.distributions.ExpRelaxedOneHotCategorical(self.discrete_voxel_vae.temperature,
                                                                  logits=latent_logits_3d)
        log_prob_q_3d = q_dist_3d.log_prob(log_token_samples_onehot_3d)  # num_samples, batch, H3, W3, D3

        prior_dist_2d = tfp.distributions.ExpRelaxedOneHotCategorical(self.temperature, logits=prior_latent_logits_2d)
        prior_dist_3d = tfp.distributions.ExpRelaxedOneHotCategorical(self.temperature, logits=prior_latent_logits_3d)
        log_prob_prior_2d = prior_dist_2d.log_prob(log_token_samples_onehot_2d)
        log_prob_prior_3d = prior_dist_3d.log_prob(log_token_samples_onehot_3d)

        kl_term_2d = tf.reduce_sum(log_prob_q_2d - log_prob_prior_2d, axis=[-1, -2])
        kl_term_3d = tf.reduce_sum(log_prob_q_3d - log_prob_prior_3d, axis=[-1, -2, -3])

        return kl_term_2d, kl_term_3d, log_prob_prior_2d, log_prob_prior_3d

    def compute_prior_logits(self, token_samples_onehot_2d, token_samples_onehot_3d):
        prior_latent_logits = self.compute_logits(token_samples_onehot_2d,
                                                  token_samples_onehot_3d)  # num_samples, batch, H2*W2 + H3*W3*D3, num_embedding2+num_embedding3
        _, batch, H2, W2, _ = get_shape(token_samples_onehot_2d)
        _, batch, H3, W3, D3, _ = get_shape(token_samples_onehot_3d)
        prior_latent_logits_2d = tf.reshape(prior_latent_logits[:, :, :H2 * W2, :self.discrete_image_vae.num_embedding],
                                            (self.num_token_samples, batch, H2, W2,
                                             self.discrete_image_vae.num_embedding))
        prior_latent_logits_3d = tf.reshape(prior_latent_logits[:, :, H2 * W2:, self.discrete_image_vae.num_embedding:],
                                            (self.num_token_samples, batch, H3, W3, D3,
                                             self.discrete_voxel_vae.num_embedding))
        return prior_latent_logits_2d, prior_latent_logits_3d

    def compute_logits(self, token_samples_onehot_2d, token_samples_onehot_3d):
        """

        Args:
            latent_tokens_2d: [num_samples, batch, H2, W2, num_embedding2]
            latent_tokens_3d: [num_samples, H3, W3, D3, num_embedding3]

        Returns:
            latent_logits:  [batch, H2*W2 + H3*W3*D3, num_embedding2 + num_embedding3]
        """

        _, batch, H2, W2, _ = get_shape(token_samples_onehot_2d)
        _, batch, H3, W3, D3, _ = get_shape(token_samples_onehot_3d)
        token_samples_onehot_2d = tf.reshape(token_samples_onehot_2d, (
            self.num_token_samples * batch, H2 * W2, self.discrete_image_vae.num_embedding))
        token_samples_onehot_3d = tf.reshape(token_samples_onehot_3d, (
            self.num_token_samples * batch, H3 * W3 * D3, self.discrete_voxel_vae.num_embedding))

        # num_samples*batch, H2*W2, num_embedding2
        latent_tokens_2d = tf.einsum("bne,ed->bnd", token_samples_onehot_2d,
                                     self.embeddings[:self.discrete_image_vae.num_embedding, :])
        # num_samples*batch, H3*W3*D3, num_embedding3
        latent_tokens_3d = tf.einsum("bne,ed->bnd", token_samples_onehot_3d,
                                     self.embeddings[self.discrete_image_vae.num_embedding:, :])

        latent_graphs = self.forced_teacher_model(latent_tokens_2d, latent_tokens_3d)

        latent_graphs = graph_batch_reshape(latent_graphs)
        latent_logits = latent_graphs.nodes  # batch, H2*W2 + H3 * W3*D3, num_embedding

        latent_logits = tf.reshape(latent_logits, (self.num_token_samples, batch, H2 * W2 + H3 * W3 * D3,
                                                   self.num_embedding))  # batch, H2*W2 + H3 * W3*D3, num_embedding
        latent_logits /= tf.math.reduce_std(latent_logits, axis=-1, keepdims=True)
        latent_logits -= tf.reduce_logsumexp(latent_logits, axis=-1, keepdims=True)
        return latent_logits

    def forced_teacher_model(self, latent_tokens_2d, latent_tokens_3d):
        """
        Args:
            latent_tokens_2d: [num_samples*batch, H2*W2, embedding_dim]
            latent_tokens_3d: [num_samples*batch, H3*W3*D3, embedding_dim]

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
            latent_tokens_2d: [num_samples*batch, H2*W2, embedding_dim]
            latent_tokens_3d: [num_samples*batch, H3*W3*D3, embedding_dim]

        Returns:
            concat_graphs: GraphsTuple, nodes are [n_graphs * (num_input + num_output), embedding_size]
        """
        G, N2, _ = get_shape(latent_tokens_2d)
        G, N3, _ = get_shape(latent_tokens_3d)
        N = N2 + N3
        nodes = tf.concat([latent_tokens_2d,
                           # tf.tile(self.starting_node_variable[None, None, :], [G, 1, 1]),
                           latent_tokens_3d], axis=1)

        self.initialize_positional_encodings(nodes)
        nodes = nodes + self.positional_encodings
        n_node = tf.fill([G], N)
        n_edge = tf.zeros_like(n_node)
        data_dict = dict(nodes=nodes, edges=None, senders=None, receivers=None, globals=None,
                         n_node=n_node, n_edge=n_edge)
        concat_graphs = GraphsTuple(**data_dict)
        concat_graphs = graph_unbatch_reshape(concat_graphs)  # [n_graphs * (num_input + num_output), embedding_size]
        # nodes, senders, receivers, globals
        concat_graphs = autoregressive_connect_graph_dynamic(concat_graphs, exclude_self_edges=True)
        return concat_graphs

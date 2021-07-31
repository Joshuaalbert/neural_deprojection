from graph_nets import blocks
import tensorflow as tf
import sonnet as snt
from graph_nets.graphs import GraphsTuple
from neural_deprojection.graph_net_utils import AbstractModule, get_shape, graph_batch_reshape, graph_unbatch_reshape, \
    grid_graphs
from neural_deprojection.models.Simple_complete_model.autoencoder_3d import DiscreteVoxelsVAE
from neural_deprojection.models.Simple_complete_model.autoencoder_2d import DiscreteImageVAE
import tensorflow_probability as tfp
from graph_nets.modules import SelfAttention
from sonnet.src import utils, once

from neural_deprojection.models.Simple_complete_model.edge_connectors import autoregressive_connect_graph_dynamic


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
        self.ln_keys = snt.LayerNorm(axis=-1, eps=1e-6, create_scale=True, create_offset=True, name='layer_norm_keys')
        self.ln_queries = snt.LayerNorm(axis=-1, eps=1e-6, create_scale=True, create_offset=True,
                                        name='layer_norm_queries')
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
        n_node, _ = get_shape(latent.nodes)
        node_values = self.v_linear(latent.nodes)
        node_keys = self.k_linear(latent.nodes)
        node_queries = self.q_linear(latent.nodes)  # n_node, num_head, F

        node_keys = self.ln_keys(node_keys)
        node_queries = self.ln_queries(node_queries)
        _, _, d_k = get_shape(node_keys)
        node_queries /= tf.math.sqrt(tf.cast(d_k, node_queries.dtype))  # n_node, F

        attended_latent = self.self_attention(node_values=node_values,
                                              node_keys=node_keys,
                                              node_queries=node_queries,
                                              attention_graph=latent)
        # n_nodes, heads, output_size -> n_nodes, heads*output_size
        output_nodes = tf.reshape(attended_latent.nodes, (n_node, self.num_heads * self.input_node_size))
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
        self.layer_norm1 = snt.LayerNorm(-1, True, True, name='layer_norm_edges')
        self.layer_norm2 = snt.LayerNorm(-1, True, True, name='layer_norm_nodes')
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
                                           use_receiver_nodes=False,
                                           use_sender_nodes=True,
                                           use_globals=self.use_globals)

        self.node_block = blocks.NodeBlock(node_model_fn,
                                           use_received_edges=True,
                                           use_sent_edges=False,
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

    Minimizes KL(q(z_2d, z_3d | x_2d, x_3d) || p(z_2d, z_3d)) for p(z_2d, z_3d) which is an auto-regressive model.
    """

    def __init__(self,
                 discrete_image_vae: DiscreteImageVAE,
                 discrete_voxel_vae: DiscreteVoxelsVAE,
                 num_heads: int = 1,
                 num_layers: int = 1,
                 embedding_dim: int = 16,
                 num_token_samples: int = 1,
                 name=None):
        super(AutoRegressivePrior, self).__init__(name=name)
        self.discrete_image_vae = discrete_image_vae
        self.discrete_voxel_vae = discrete_voxel_vae
        self.embedding_dim = embedding_dim
        self.num_token_samples = num_token_samples
        self.num_embedding = self.discrete_voxel_vae.num_embedding + self.discrete_image_vae.num_embedding + 3

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

    def _core(self, graphs):
        return self.selfattention_core(graphs)

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
        return self._deproject_images(images)

    def _deproject_images(self, images):
        """
        For a batch of images samples a 3D medium consistent with the image.

        Args:
            images: [batch, H2, W2, C2]

        Returns:
            mu, b: mean, uncertainty of images [batch, H3, W2, D3, C3]
        """
        logits_2d = self.discrete_image_vae.compute_logits(images)  # [batch, W2, H2, num_embeddings2]
        dist_2d = tfp.distributions.Categorical(logits=logits_2d, dtype=tf.int32)
        token_samples_idx_2d = dist_2d.sample(1)  # [1, batch, W2, H2]
        token_samples_idx_2d = token_samples_idx_2d[0]  # batch, W2, H2
        token_samples_idx_3d = self._incrementally_decode(token_samples_idx_2d)  # batch, H3,W3,D3
        latent_token_samples_3d = tf.nn.embedding_lookup(self.discrete_voxel_vae.embeddings, token_samples_idx_3d)# [batch, W3, H3, D3, embedding_size3]
        mu_3d, logb_3d = self.discrete_voxel_vae.compute_likelihood_parameters(
            latent_token_samples_3d[None])  # [1, batch, H', W', D' C], [1, batch, H', W', D' C]_
        mu_3d = mu_3d[0]
        logb_3d = logb_3d[0]
        b_3d = tf.math.exp(logb_3d)
        return mu_3d, b_3d

    def _incrementally_decode(self, token_samples_idx_2d):
        """
        Args:
            token_samples_idx_2d: [batch, H2, W2]

        Returns:
            token_samples_idx_3d: [batch, H3,W3,D3]
        """
        idx_dtype = token_samples_idx_2d.dtype

        batch, H2, W2 = get_shape(token_samples_idx_2d)
        H3 = W3 = D3 = self.discrete_voxel_vae.voxels_per_dimension // self.discrete_voxel_vae.shrink_factor

        token_samples_idx_2d = tf.reshape(token_samples_idx_2d, (batch, H2 * W2))

        N, _ = get_shape(self.positional_encodings)

        # batch, H3*W3*D3, num_embedding3
        token_samples_idx_3d = tf.zeros((batch, H3 * W3 * D3), dtype=idx_dtype)

        def _core(output_token_idx, token_samples_idx_3d):
            """

            Args:
                output_token_idx: which element is being replaced
                token_samples_idx_3d: [batch, H3*W3*D3]
            """
            # [batch, 1 + H2*W2 + 1 + H3*W3*D3 + 1]
            sequence = self.construct_sequence(token_samples_idx_2d, token_samples_idx_3d)
            input_sequence = sequence[:, :-1]
            input_graphs = self.construct_input_graph(input_sequence)
            latent_logits = self.compute_logits(input_graphs)

            #batch, H3 * W3 * D3, num_embedding3
            # . a b . c d
            # a b . c d .
            prior_latent_logits_3d = latent_logits[:, H2*W2+1:H2*W2+1+H3*W3*D3,
                                                self.discrete_image_vae.num_embedding:self.discrete_image_vae.num_embedding + self.discrete_voxel_vae.num_embedding]

            prior_dist = tfp.distributions.Categorical(logits=prior_latent_logits_3d, dtype=idx_dtype)
            prior_latent_tokens_idx_3d = prior_dist.sample(1)[0] # batch, H3*W3*D3
            # import pylab as plt
            # # plt.imshow(tf.one_hot(prior_latent_tokens_idx_3d[0, :30], self.discrete_voxel_vae.num_embedding))
            # plt.imshow(latent_logits[0, 1020:1050], aspect='auto', interpolation='nearest')
            # plt.show()

            _mask = tf.range(H3 * W3 * D3) == output_token_idx  # [H3*W3*D3]

            output_token_samples_idx_3d = tf.where(_mask[None, :],
                                                      prior_latent_tokens_idx_3d,
                                                      token_samples_idx_3d
                                                      )

            return (output_token_idx + 1, output_token_samples_idx_3d)

        _, token_samples_idx_3d = tf.while_loop(
            cond=lambda output_token_idx, _: output_token_idx < (H3 * W3 * D3),
            body=_core,
            loop_vars=(tf.convert_to_tensor(0), token_samples_idx_3d))

        # latent_graphs = GraphsTuple(**latent_graphs_data_dict, edges=None, globals=None)

        token_samples_idx_3d = tf.reshape(token_samples_idx_3d,
                                             (batch, H3, W3, D3))
        return token_samples_idx_3d

    def write_summary(self,images, graphs,
                               latent_logits_2d,
                               latent_logits_3d,
                               prior_latent_logits_2d,
                               prior_latent_logits_3d):

        dist_2d = tfp.distributions.OneHotCategorical(logits=latent_logits_2d, dtype=latent_logits_2d.dtype)
        dist_3d = tfp.distributions.OneHotCategorical(logits=latent_logits_3d, dtype=latent_logits_3d.dtype)
        token_samples_onehot_2d = dist_2d.sample(1)[0]
        token_samples_onehot_3d = dist_3d.sample(1)[0]

        dist_2d_prior = tfp.distributions.OneHotCategorical(logits=prior_latent_logits_2d, dtype=prior_latent_logits_2d.dtype)
        dist_3d_prior = tfp.distributions.OneHotCategorical(logits=prior_latent_logits_3d, dtype=prior_latent_logits_3d.dtype)
        prior_token_samples_onehot_2d = dist_2d_prior.sample(1)[0]
        prior_token_samples_onehot_3d = dist_3d_prior.sample(1)[0]

        kl_div_2d = tf.reduce_mean(tf.reduce_sum(dist_2d.kl_divergence(dist_2d_prior), axis=[-1,-2]))
        kl_div_3d = tf.reduce_mean(tf.reduce_sum(dist_3d.kl_divergence(dist_3d_prior), axis=[-1,-2,-3]))
        tf.summary.scalar('kl_div_2d', kl_div_2d, step=self.step)
        tf.summary.scalar('kl_div_3d', kl_div_3d, step=self.step)
        tf.summary.scalar('kl_div', kl_div_2d + kl_div_3d, step=self.step)

        perplexity_2d = 2. ** (dist_2d_prior.entropy() / tf.math.log(2.))  #
        mean_perplexity_2d = tf.reduce_mean(perplexity_2d)  # scalar

        perplexity_3d = 2. ** (dist_3d_prior.entropy() / tf.math.log(2.))  #
        mean_perplexity_3d = tf.reduce_mean(perplexity_3d)  #

        tf.summary.scalar('perplexity_2d_prior', mean_perplexity_2d, step=self.step)
        tf.summary.scalar('perplexity_3d_prior', mean_perplexity_3d, step=self.step)

        prior_latent_tokens_2d = tf.einsum('sbhwd,de->sbhwe', prior_token_samples_onehot_2d[None], self.discrete_image_vae.embeddings)
        prior_latent_tokens_3d = tf.einsum('sbhwdn,ne->sbhwde', prior_token_samples_onehot_3d[None], self.discrete_voxel_vae.embeddings)
        mu_2d, logb_2d = self.discrete_image_vae.compute_likelihood_parameters(
            prior_latent_tokens_2d)  # [num_samples, batch, H', W', C], [num_samples, batch, H', W', C]
        log_likelihood_2d = self.discrete_image_vae.log_likelihood(images, mu_2d, logb_2d)  # [num_samples, batch]
        var_exp_2d = tf.reduce_mean(log_likelihood_2d)  # [scalar]
        mu_3d, logb_3d = self.discrete_voxel_vae.compute_likelihood_parameters(
            prior_latent_tokens_3d)  # [num_samples, batch, H', W', D', C], [num_samples, batch, H', W', D', C]
        log_likelihood_3d = self.discrete_voxel_vae.log_likelihood(graphs, mu_3d, logb_3d)  # [num_samples, batch]
        var_exp_3d = tf.reduce_mean(log_likelihood_3d)  # [scalar]
        var_exp = log_likelihood_2d + log_likelihood_3d

        tf.summary.scalar('var_exp_3d', tf.reduce_mean(var_exp_3d), step=self.step)
        tf.summary.scalar('var_exp_2d', tf.reduce_mean(var_exp_2d), step=self.step)
        tf.summary.scalar('var_exp', tf.reduce_mean(var_exp), step=self.step)


        projected_mu = tf.reduce_sum(mu_3d[0], axis=-2)  # [batch, H', W', C]
        voxels = grid_graphs(graphs, self.discrete_voxel_vae.voxels_per_dimension)  # [batch, H', W', D', C]
        projected_img = tf.reduce_sum(voxels, axis=-2)  # [batch, H', W', C]
        for i in range(self.discrete_voxel_vae.num_channels):
            vmin = tf.reduce_min(projected_mu[..., i])
            vmax = tf.reduce_max(projected_mu[..., i])
            _projected_mu = (projected_mu[..., i:i + 1] - vmin) / (vmax - vmin)  # batch, H', W', 1
            _projected_mu = tf.clip_by_value(_projected_mu, 0., 1.)

            vmin = tf.reduce_min(projected_img[..., i])
            vmax = tf.reduce_max(projected_img[..., i])
            _projected_img = (projected_img[..., i:i + 1] - vmin) / (vmax - vmin)  # batch, H', W', 1
            _projected_img = tf.clip_by_value(_projected_img, 0., 1.)

            tf.summary.image(f'voxels_predict_prior[{i}]', _projected_mu, step=self.step)
            tf.summary.image(f'voxels_actual[{i}]', _projected_img, step=self.step)

        for name, _latent_logits_3d, _tokens_onehot_3d in zip(['', '_prior'],
                                          [latent_logits_3d,prior_latent_logits_3d],
                                                              [token_samples_onehot_3d, prior_token_samples_onehot_3d]):
            batch, H3, W3, D3, _ = get_shape(_latent_logits_3d)
            _latent_logits_3d -= tf.reduce_min(_latent_logits_3d, axis=-1, keepdims=True)
            _latent_logits_3d /= tf.reduce_max(_latent_logits_3d, axis=-1, keepdims=True)
            _latent_logits_3d = tf.reshape(_latent_logits_3d,
                                           [batch, H3 * W3 * D3, self.discrete_voxel_vae.num_embedding,
                                            1])  # [batch, H*W*D, num_embedding, 1]
            tf.summary.image(f"latent_logits_3d{name}", _latent_logits_3d, step=self.step)

            _tokens_onehot_3d = tf.reshape(_tokens_onehot_3d,
                                                [batch, H3 * W3 * D3, self.discrete_voxel_vae.num_embedding,
                                                 1])  # [batch, H*W*D, num_embedding, 1]
            tf.summary.image(f'latent_samples_onehot_3d{name}', _tokens_onehot_3d, step=self.step)

        _mu = mu_2d[0]  # [batch, H', W', C]
        _img = images  # [batch, H', W', C]
        for i in range(self.discrete_image_vae.num_channels):
            vmin = tf.reduce_min(_mu[..., i])
            vmax = tf.reduce_max(_mu[..., i])
            _projected_mu = (_mu[..., i:i + 1] - vmin) / (vmax - vmin)  # batch, H', W', 1
            _projected_mu = tf.clip_by_value(_projected_mu, 0., 1.)

            vmin = tf.reduce_min(_img[..., i])
            vmax = tf.reduce_max(_img[..., i])
            _projected_img = (_img[..., i:i + 1] - vmin) / (vmax - vmin)  # batch, H', W', 1
            _projected_img = tf.clip_by_value(_projected_img, 0., 1.)

            tf.summary.image(f'image_predict_prior[{i}]', _projected_mu, step=self.step)
            tf.summary.image(f'image_actual[{i}]', _projected_img, step=self.step)

        for name, _latent_logits_2d, _tokens_onehot_2d in zip(['', '_prior'],
                                                              [latent_logits_2d, prior_latent_logits_2d],
                                                              [token_samples_onehot_2d, prior_token_samples_onehot_2d]):
            batch, H2, W2, _ = get_shape(_latent_logits_2d)
            _latent_logits_2d -= tf.reduce_min(_latent_logits_2d, axis=-1, keepdims=True)
            _latent_logits_2d /= tf.reduce_max(_latent_logits_2d, axis=-1, keepdims=True)
            _latent_logits_2d = tf.reshape(_latent_logits_2d,
                                           [batch, H2 * W2, self.discrete_image_vae.num_embedding,
                                            1])  # [batch, H*W*D, num_embedding, 1]
            tf.summary.image(f"latent_logits_2d{name}", _latent_logits_2d, step=self.step)

            _tokens_onehot_2d = tf.reshape(_tokens_onehot_2d,
                                           [batch, H2 * W2, self.discrete_image_vae.num_embedding,
                                            1])  # [batch, H*W*D, num_embedding, 1]
            tf.summary.image(f'latent_samples_onehot_2d{name}', _tokens_onehot_2d, step=self.step)

    def _build(self, graphs, images):

        idx_dtype = tf.int32

        latent_logits_2d = self.discrete_image_vae.compute_logits(images)  # [batch, H, W, num_embeddings]
        latent_logits_3d = self.discrete_voxel_vae.compute_logits(graphs)  # [batch, H, W, D, num_embeddings]

        batch, H2, W2, _ = get_shape(latent_logits_2d)
        batch, H3, W3, D3, _ = get_shape(latent_logits_3d)
        G = self.num_token_samples * batch

        latent_logits_2d = tf.reshape(latent_logits_2d, (batch, H2 * W2, self.discrete_image_vae.num_embedding))
        latent_logits_3d = tf.reshape(latent_logits_3d, (batch, H3 * W3 * D3, self.discrete_voxel_vae.num_embedding))

        q_dist_2d = tfp.distributions.Categorical(logits=latent_logits_2d, dtype=idx_dtype)
        token_samples_idx_2d = q_dist_2d.sample(self.num_token_samples)  # [num_samples, batch, H2*W2]
        token_samples_idx_2d = tf.reshape(token_samples_idx_2d, (G, H2*W2))

        q_dist_3d = tfp.distributions.Categorical(logits=latent_logits_3d, dtype=idx_dtype)
        token_samples_idx_3d = q_dist_3d.sample(
            self.num_token_samples)  # [num_samples, batch, H3*W3*D3]
        token_samples_idx_3d = tf.reshape(token_samples_idx_3d, (G, H3*W3*D3))

        entropy_2d = tf.reduce_sum(q_dist_2d.entropy(), axis=-1)
        entropy_3d = tf.reduce_sum(q_dist_3d.entropy(), axis=-1)
        entropy = entropy_3d + entropy_2d  # [batch]
        ## create sequence

        sequence = self.construct_sequence(token_samples_idx_2d, token_samples_idx_3d)
        input_sequence = sequence[:, :-1]
        input_graphs = self.construct_input_graph(input_sequence)
        latent_logits = self.compute_logits(input_graphs)

        prior_dist = tfp.distributions.Categorical(logits=latent_logits, dtype=idx_dtype)
        output_sequence = sequence[:, 1:]
        cross_entropy = -prior_dist.log_prob(output_sequence)#num_samples*batch, H2*W2+1+H3*W3*D3+1
        # . a . b
        # a . b .
        cross_entropy = cross_entropy[:, H2*W2+1:-1]
        cross_entropy = tf.reshape(tf.reduce_sum(cross_entropy, axis=-1), (self.num_token_samples, batch))  # num_samples,batch
        kl_term = cross_entropy + entropy  # [num_samples, batch]

        kl_div = tf.reduce_mean(kl_term)  # scalar
        # elbo = tf.stop_gradient(var_exp) - self.beta * kl_div
        elbo = - kl_div  # scalar

        loss = - elbo  # scalar

        if self.step % 100 == 0:
            prior_latent_logits_2d = tf.reshape(latent_logits[:, :H2*W2, :self.discrete_image_vae.num_embedding],
                                                (self.num_token_samples, batch, H2, W2, self.discrete_image_vae.num_embedding))
            prior_latent_logits_3d = tf.reshape(latent_logits[:, H2*W2+1:H2*W2+1+H3*W3*D3,
                                     self.discrete_image_vae.num_embedding:self.discrete_image_vae.num_embedding+self.discrete_voxel_vae.num_embedding],
                                                (self.num_token_samples, batch, H3, W3, D3, self.discrete_voxel_vae.num_embedding))

            latent_logits_2d = tf.reshape(latent_logits_2d,
                                                (batch, H2, W2, self.discrete_image_vae.num_embedding))
            latent_logits_3d = tf.reshape(latent_logits_3d,
                                          (batch, H3, W3, D3, self.discrete_voxel_vae.num_embedding))

            self.write_summary(images, graphs,
                               latent_logits_2d,
                               latent_logits_3d,
                               prior_latent_logits_2d[0],
                               prior_latent_logits_3d[0])

        return dict(loss=loss)

    def compute_logits(self, input_graphs):
        latent_graphs = self._core(input_graphs)
        latent_graphs = graph_batch_reshape(latent_graphs)
        latent_logits = latent_graphs.nodes  # num_samples*batch, H2*W2 + 1 + H3*W3*D3 + 1, num_embedding
        latent_logits /= 1e-6 + tf.math.reduce_std(latent_logits, axis=-1, keepdims=True)
        latent_logits -= tf.reduce_logsumexp(latent_logits, axis=-1, keepdims=True)
        return latent_logits

    def construct_input_graph(self, input_sequence):
        G, N = get_shape(input_sequence)
        # num_samples*batch, 1 + H2*W2 + 1 + H3*W3*D3, embedding_dim
        input_tokens = tf.nn.embedding_lookup(self.embeddings, input_sequence)
        self.initialize_positional_encodings(input_tokens)
        nodes = input_tokens + self.positional_encodings
        n_node = tf.fill([G], N)
        n_edge = tf.zeros_like(n_node)
        data_dict = dict(nodes=nodes, edges=None, senders=None, receivers=None, globals=None,
                         n_node=n_node,
                         n_edge=n_edge)
        concat_graphs = GraphsTuple(**data_dict)
        concat_graphs = graph_unbatch_reshape(concat_graphs)  # [n_graphs * (num_input + num_output), embedding_size]
        # nodes, senders, receivers, globals
        concat_graphs = autoregressive_connect_graph_dynamic(concat_graphs, exclude_self_edges=False)
        return concat_graphs

    def construct_sequence(self, token_samples_idx_2d, token_samples_idx_3d):
        """

        Args:
            token_samples_idx_2d: [G, H2*W2]
            token_samples_idx_3d: [G, H3*W3*D3]

        Returns:
            sequence: G, 1 + H2*W2 + 1 + H3*W3*D3 + 1
        """
        idx_dtype = token_samples_idx_2d.dtype
        G, N2 = get_shape(token_samples_idx_2d)
        G, N3 = get_shape(token_samples_idx_3d)
        start_token_idx = tf.constant(self.num_embedding - 3, dtype=idx_dtype)
        del_token_idx = tf.constant(self.num_embedding - 2, dtype=idx_dtype)
        eos_token_idx = tf.constant(self.num_embedding - 1, dtype=idx_dtype)
        start_token = tf.fill((G, 1), start_token_idx)
        del_token = tf.fill((G, 1), del_token_idx)
        eos_token = tf.fill((G, 1), eos_token_idx)
        ###
        # num_samples*batch, 1 + H2*W2 + 1 + H3*W3*D3 + 1
        sequence = tf.concat([
            start_token,
            token_samples_idx_2d,
            del_token,
            token_samples_idx_3d + self.discrete_image_vae.num_embedding,  # shift to right
            eos_token
        ], axis=-1)
        return sequence

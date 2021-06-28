import tensorflow as tf
import sonnet as snt
from graph_nets.graphs import GraphsTuple
from graph_nets.modules import SelfAttention
from neural_deprojection.graph_net_utils import AbstractModule, get_shape, graph_batch_reshape, graph_unbatch_reshape, histogramdd
from neural_deprojection.models.Simple_complete_model.graph_decoder import GraphMappingNetwork
from sonnet.src import utils, once


class SimpleCompleteModel(AbstractModule):
    def __init__(self,
                 num_properties: int,
                 num_components:int,
                 component_size:int,
                 num_embedding_3d:int,
                 edge_size:int,
                 global_size:int,
                 n_node_per_graph:int,
                 discrete_image_vae,
                 num_token_samples: int,
                 multi_head_output_size: int,
                 num_heads: int,
                 batch: int,
                 name=None):
        super(SimpleCompleteModel, self).__init__(name=name)

        self.discrete_image_vae = discrete_image_vae
        self.decoder = GraphMappingNetwork(num_output=num_components,
                                           num_embedding=num_embedding_3d,
                                           multi_head_output_size=multi_head_output_size,
                                           num_heads=num_heads,
                                           embedding_size=component_size,
                                           edge_size=edge_size,
                                           global_size=global_size)
        self.num_token_samples = num_token_samples
        self.num_properties = num_properties
        self.num_components = num_components
        self.component_size = component_size
        self.n_node_per_graph = n_node_per_graph
        self.batch = batch
        self.temperature = tf.Variable(initial_value=tf.constant(1.), name='temperature', trainable=False)
        self.beta = tf.Variable(initial_value=tf.constant(1.), name='beta', trainable=False)

        # output is [mu, log_stddev]
        self.field_reconstruction = snt.nets.MLP([num_properties * 10 * 2, num_properties * 10 * 2, 2*num_properties],
                                                 activate_final=False)

    def encoder_2d(self, img):
        latent_logits = self.discrete_image_vae.sample_encoder(img)
        return latent_logits

    def sample_latent_2d(self, latent_logits, temperature, num_token_samples):
        token_samples_onehot, token_samples = self.discrete_image_vae.sample_latent_2d(latent_logits,
                                                                                       temperature,
                                                                                       num_token_samples)
        return token_samples_onehot, token_samples

    def reconstruct_field(self, field_component_tokens, positions, basis_weights):
        """
        Reconstruct the field at positions.

        Args:
            field_component_tokens: [num_token_samples, batch, output_size, embedding_dim_3d]
            positions: [num_token_samples, batch, n_node, 3]
            basis_weights: [num_token_samples, batch, output_size]

        Returns:
            [num_token_samples, batch, n_node, num_properties*2]
        """
        pos_shape = get_shape(positions)

        def _single_sample(args):
            """
            Compute for single batch.

            Args:
                tokens: [batch, output_size, embedding_dim_3d]
                positions: [batch, n_node, 3]
                basis_weights: [batch, output_size]

            Returns:
                [n_node, num_properties]
            """
            tokens, positions, basis_weights = args

            def _single_batch(args):
                """
                Compute for single batch.

                Args:
                    tokens: [output_size, embedding_dim_3d]
                    positions: [n_node, 3]
                    basis_weights: [output_size]

                Returns:
                    [n_node, num_properties]
                """
                tokens, positions, basis_weights = args
                #output_size, num_properties*2
                basis_weights = tf.concat([tf.tile(basis_weights[:, None], [1, self.num_properties]),
                                           tf.ones([self.num_components, self.num_properties])], axis=-1)

                def _single_component(token):
                    """
                    Compute for a single component.

                    Args:
                        token: [component_size]

                    Returns:
                        [n_node, num_properties]
                    """

                    features = tf.concat([positions, tf.tile(token[None, :], [pos_shape[2], 1])], axis=-1)  # [n_node, 3 + embedding_dim_3D]
                    return self.field_reconstruction(features)  # [n_node, num_properties]

                return tf.reduce_sum(basis_weights[:, None, :] * tf.vectorized_map(_single_component, tokens), axis=0)  # [n_node, num_properties]

            return tf.vectorized_map(_single_batch, (tokens, positions))  # [batch, n_node, num_properties]

        return tf.vectorized_map(_single_sample, (field_component_tokens, positions, basis_weights))  # [num_token_samples, batch, n_node, num_properties*2]

    @tf.function(input_signature=[tf.TensorSpec([None, None, None, None], dtype=tf.float32),
                                  tf.TensorSpec([None, 3], dtype=tf.float32),
                                  tf.TensorSpec([], dtype=tf.float32)])
    def im_to_components(self, im, positions, temperature):
        # returns mu, std, so return only mu
        mu, _ = self._im_to_components(im, positions, temperature)
        return mu

    def _im_to_components(self, im, positions, temperature):
        '''

        Args:
            im: [batch=1, H, W, 2*C]
            positions: [num_positions, 3]
            temperature: scalar

        Returns:

        '''
        latent_logits = self.encoder_2d(im)  # batch, H, W, num_embeddings

        [_, H, W, num_embedding] = get_shape(latent_logits)

        latent_logits = tf.reshape(latent_logits, [1, H * W, num_embedding])  # [batch, H*W, num_embeddings]
        reduce_logsumexp = tf.math.reduce_logsumexp(latent_logits, axis=-1)  # [batch, H*W]
        reduce_logsumexp = tf.tile(reduce_logsumexp[..., None], [1, 1, num_embedding])  # [batch, H*W, num_embedding]
        latent_logits -= reduce_logsumexp  # [batch, H*W, num_embeddings]

        token_samples_onehot, token_samples = self.sample_latent_2d(latent_logits,
                                                                    temperature,
                                                                    1)  # [num_token_samples, batch, H*W, num_embedding / embedding_dim]

        [_, _, _, embedding_dim] = get_shape(token_samples)

        token_samples = tf.reshape(token_samples, [self.n_node_per_graph,
                                                   embedding_dim])  # [num_token_samples * batch * H * W, embedding_dim]

        token_graphs = GraphsTuple(nodes=token_samples,
                                   edges=None,
                                   globals=None,
                                   senders=None,
                                   receivers=None,
                                   n_node=tf.constant([self.n_node_per_graph],dtype=tf.int32),
                                   n_edge=tf.constant([0],
                                                      dtype=tf.int32))  # [n_node, embedding_dim], n_node = n_graphs * n_node_per_graph

        tokens_3d, kl_div, token_3d_samples_onehot, basis_weights = self.decoder(token_graphs,
                                                                  temperature)  # [n_graphs, num_output, embedding_dim_3d], [n_graphs], [n_graphs, num_output, num_embedding_3d], [n_graphs, num_output]
        tokens_3d = tf.reshape(tokens_3d, [1,1,self.num_components,
                                           self.component_size])  # [num_token_samples, batch, num_output, embedding_dim_3d]
        basis_weights = tf.reshape(basis_weights, [1, 1, self.num_components])
        #take off the fake sample and batch dims
        log_likelihood_params = self.reconstruct_field(tokens_3d, positions, basis_weights)[0,0,:,:]
        mu = log_likelihood_params[..., :self.num_properties]
        log_stddev = log_likelihood_params[..., self.num_properties:]
        return mu, log_stddev


    def set_beta(self, beta):
        self.beta.assign(beta)

    def set_temperature(self, temperature):
        self.temperature.assign(temperature)

    def log_likelihood(self, tokens_3d, properties, basis_weights):
        """
                Args:
                    tokens_3d: [num_token_samples, batch, n_tokens, embedding_dim_3d]
                    properties: [num_token_samples, batch, n_node_per_graph, 3 + num_properties]
                    basis_weights: [num_token_samples, batch, n_node_per_graph]
                Returns:
                    scalar
                """

        positions = properties[:, :, :, :3]  # [num_token_samples, batch, n_node_per_graph, 3]
        input_properties = properties[:, :, :, 3:]  # [num_token_samples, batch, n_node_per_graph, num_properties]

        mu_field_properties, log_stddev_field_properties = self.reconstruct_field(tokens_3d, positions, basis_weights)  # [num_token_samples, batch, n_node_per_graph, num_properties]
        # todo: add variance (reconstruct_field would return it)
        diff_properties = (input_properties - mu_field_properties)   # [num_token_samples, batch, n_node_per_graph, num_properties]
        diff_properties /= tf.math.exp(log_stddev_field_properties)
        maha_term = -0.5 * tf.math.square(diff_properties)   #[num_token_samples, batch, n_node_per_graph, num_properties]
        log_det_term = -log_stddev_field_properties #[num_token_samples, batch, n_node_per_graph, num_properties]
        log_likelihood = tf.reduce_mean(tf.reduce_sum(maha_term + log_det_term, axis=-1), axis=-1) #num_token_samples, batch]

        return mu_field_properties, log_likelihood# [num_token_samples, batch, n_node_per_graph, num_properties], [num_token_samples, batch]



    def _build(self, graphs, imgs, **kwargs) -> dict:

        # graphs.nodes: [batch, n_node_per_graph, 3+num_properties]
        # imgs: [batch, H', W', C]
        latent_logits = self.encoder_2d(imgs) #batch, H, W, num_embeddings

        [_, H, W, num_embedding] = get_shape(latent_logits)

        latent_logits = tf.reshape(latent_logits, [self.batch, H*W, num_embedding])  # [batch, H*W, num_embeddings]
        reduce_logsumexp = tf.math.reduce_logsumexp(latent_logits, axis=-1)  # [batch, H*W]
        reduce_logsumexp = tf.tile(reduce_logsumexp[..., None], [1, 1, num_embedding]) # [batch, H*W, num_embedding]
        latent_logits -= reduce_logsumexp  # [batch, H*W, num_embeddings]

        # temperature = tf.maximum(0.1, tf.cast(10. - 0.1 * (self.step / 1000), tf.float32))
        # temperature = 10.
        token_samples_onehot, token_samples = self.sample_latent_2d(latent_logits,
                                                                    self.temperature,
                                                                    self.num_token_samples) # [num_token_samples, batch, H*W, num_embedding / embedding_dim]

        # token_samples = tf.reshape(token_samples, [self.num_token_samples * self.batch, H * W, self.component_size])  # [num_token_samples*batch, H*W, embedding_dim]
        [_, _, _, embedding_dim] = get_shape(token_samples)
        n_graphs = self.num_token_samples * self.batch

        token_samples = tf.reshape(token_samples, [n_graphs * self.n_node_per_graph, embedding_dim])  # [num_token_samples * batch * H * W, embedding_dim]

        token_graphs = GraphsTuple(nodes=token_samples,
                                   edges=None,
                                   globals=None,
                                   senders=None,
                                   receivers=None,
                                   n_node=tf.constant(n_graphs * [self.n_node_per_graph], dtype=tf.int32),
                                   n_edge=tf.constant(n_graphs * [0], dtype=tf.int32))  # [n_node, embedding_dim], n_node = n_graphs * n_node_per_graph

        tokens_3d, kl_div, token_3d_samples_onehot, basis_weights = self.decoder(token_graphs, self.temperature)  # [n_graphs, num_output, embedding_dim_3d], [n_graphs], [n_graphs, num_output, num_embedding_3d], [n_graphs, num_output]
        tokens_3d = tf.reshape(tokens_3d, [self.num_token_samples,
                                           self.batch,
                                           self.num_components,
                                           self.component_size])  # [num_token_samples, batch, num_output, embedding_dim_3d]
        basis_weights = tf.reshape(basis_weights, [self.num_token_samples,
                                           self.batch,
                                           self.num_components]) # [num_token_samples, batch, num_output]
        kl_div = tf.reshape(kl_div, [self.num_token_samples,
                                     self.batch])  # [num_token_samples, batch]

        # properties: [num_token_samples, batch, n_node_per_graph, 3+num_properties]
        # tokens_3d: [num_token_samples, batch, num_output, embedding_dim_3d]
        properties = tf.tile(graphs.nodes[None, ...], [self.num_token_samples, 1, 1, 1])
        field_properties, log_likelihood = self.log_likelihood(tokens_3d, properties, basis_weights)
        # field_properties: [num_token_samples, batch, n_node_per_graph, num_properties]
        # log_likelihood: [num_token_samples, batch]

        field_properties = tf.reduce_mean(field_properties, axis=0)  # [batch, n_node_per_graph, num_properties]
        var_exp = tf.reduce_mean(log_likelihood, axis=0)  # [batch]
        kl_div = tf.reduce_mean(kl_div, axis=0)  # [batch]
        elbo = tf.reduce_mean(var_exp - self.beta * kl_div)  # scalar

        entropy = -tf.reduce_sum(latent_logits * tf.math.exp(latent_logits), axis=-1)  # [batch, H, W]
        perplexity = 2. ** (-entropy / tf.math.log(2.))
        mean_perplexity = tf.reduce_mean(perplexity)

        loss = - elbo  # maximize ELBO so minimize -ELBO

        [_, _, _, num_channels] = get_shape(imgs)

        if self.step % 10 == 0:
            tf.summary.scalar('perplexity', mean_perplexity, step=self.step)
            tf.summary.scalar('var_exp', tf.reduce_mean(var_exp), step=self.step)
            tf.summary.scalar('kl_div', tf.reduce_mean(kl_div), step=self.step)
            tf.summary.scalar('temperature', self.temperature, step=self.step)

            tf.summary.image('token_3d_samples_onehot', token_3d_samples_onehot[..., None], step=self.step)

        if self.step % 10 == 0:
            input_properties = graphs.nodes[0]
            reconstructed_properties = field_properties[0]
            pos = tf.reverse(input_properties[:, :2], [1])

            for i in range(num_channels):
                img_i = imgs[0, ..., i][None, ..., None]
                img_i = (img_i - tf.reduce_min(img_i)) / (
                        tf.reduce_max(img_i) - tf.reduce_min(img_i))
                tf.summary.image(f'img_before_autoencoder_{i}', img_i, step=self.step)

            for i in range(self.num_properties):
                image_before, _ = histogramdd(pos, bins=64, weights=input_properties[:, 3+i])
                image_before -= tf.reduce_min(image_before)
                image_before /= tf.reduce_max(image_before)
                tf.summary.image(f"{3+i}_xy_image_before_b", image_before[None, :, :, None], step=self.step)
                tf.summary.scalar(f"properties{3+i}_std_before", tf.math.reduce_std(input_properties[:, 3+i]), step=self.step)

                image_after, _ = histogramdd(pos, bins=64, weights=reconstructed_properties[:, i])
                # image_after, _ = histogramdd(pos, bins=50, weights=tf.random.truncated_normal((10000, )))
                image_after -= tf.reduce_min(image_after)
                image_after /= tf.reduce_max(image_after)
                tf.summary.image(f"{3+i}_xy_image_after", image_after[None, :, :, None], step=self.step)
                tf.summary.scalar(f"properties{3+i}_std_after", tf.math.reduce_std(reconstructed_properties[:, i]), step=self.step)

        return dict(loss=loss,
                    metrics=dict(var_exp=var_exp,
                                 kl_term=kl_div,
                                 mean_perplexity=mean_perplexity))


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
import tensorflow as tf
import sonnet as snt
from graph_nets.graphs import GraphsTuple
from neural_deprojection.graph_net_utils import AbstractModule, get_shape, graph_batch_reshape, graph_unbatch_reshape, histogramdd
from neural_deprojection.models.Simple_complete_model.graph_decoder import GraphMappingNetwork


class SimpleCompleteModel(AbstractModule):
    def __init__(self,
                 num_properties: int,
                 num_components:int,
                 component_size:int,
                 num_embedding_3d:int,
                 edge_size:int,
                 global_size:int,
                 discrete_image_vae,
                 num_token_samples: int,
                 batch: int,
                 beta,
                 name=None):
        super(SimpleCompleteModel, self).__init__(name=name)

        self.discrete_image_vae = discrete_image_vae
        self.decoder = GraphMappingNetwork(num_output=num_components,
                                           num_embedding=num_embedding_3d,
                                           embedding_size=component_size,
                                           edge_size=edge_size,
                                           global_size=global_size)
        self.num_token_samples = num_token_samples
        self.num_properties = num_properties
        self.num_components = num_components
        self.component_size = component_size
        self.batch = batch
        self.beta = tf.Variable(tf.constant(beta, dtype=tf.float32), name='beta')

        self.field_reconstruction = snt.nets.MLP([num_properties * 10 * 2, num_properties * 10 * 2, num_properties],
                                                 activate_final=False)

    def encoder_2d(self, img):
        latent_logits = self.discrete_image_vae.sample_encoder(img)
        return latent_logits

    def sample_latent_2d(self, latent_logits, temperature, num_token_samples):
        token_samples_onehot, token_samples = self.discrete_image_vae.sample_latent_2d(latent_logits,
                                                                                       temperature,
                                                                                       num_token_samples)
        return token_samples_onehot, token_samples

    def reconstruct_field(self, field_component_tokens, positions):
        """
        Reconstruct the field at positions.

        Args:
            field_component_tokens: [num_token_samples, batch, output_size, embedding_dim_3d]
            positions: [num_token_samples, batch, n_node, 3]

        Returns:
            [batch, n_node, num_properties]
        """
        pos_shape = get_shape(positions)

        def _single_sample(args):
            """
            Compute for single batch.

            Args:
                tokens: [batch, output_size, embedding_dim_3d]
                positions: [batch, n_node, 3]

            Returns:
                [n_node, num_properties]
            """
            tokens, positions = args

            def _single_batch(args):
                """
                Compute for single batch.

                Args:
                    tokens: [output_size, embedding_dim_3d]
                    positions: [n_node, 3]

                Returns:
                    [n_node, num_properties]
                """
                tokens, positions = args

                def _single_component(token):
                    """
                    Compute for a single component.

                    Args:
                        token: [component_size]

                    Returns:
                        [n_node, num_properties]
                    """
                    # n_node, 3+num_properties*10
                    # tf.concat([n_node, 3], [3, embedding_dim_3D])
                    # concat([n_node, 3], [n_node, embedding_dim_3D]) -> [n_node, 3+embedding_dim_3D]

                    features = tf.concat([positions, tf.tile(token[None, :], [pos_shape[2], 1])], axis=-1)
                    return self.field_reconstruction(features)

                return tf.reduce_sum(tf.vectorized_map(_single_component, tokens), axis=0)

            return tf.vectorized_map(_single_batch, (tokens, positions))

        return tf.vectorized_map(_single_sample, (field_component_tokens, positions))

    def set_beta(self, beta):
        self.beta.assign(beta)

    def set_temperature(self, temperature):
        self.temperature.assign(temperature)
        
    def log_likelihood(self, tokens_3d, properties):
        """
                Args:
                    tokens_3d: [num_token_samples, batch, n_tokens, embedding_dim_3d]
                    properties: [num_token_samples, batch, n_node_per_graph, 3 + num_properties]

                Returns:
                    scalar
                """

        positions = properties[:, :, :, :3]  # [num_token_samples, batch, n_node_per_graph, 3]
        input_properties = properties[:, :, :, 3:]  # [num_token_samples, batch, n_node_per_graph, num_properties]

        field_properties = self.reconstruct_field(tokens_3d, positions)  # [num_token_samples, batch, n_node_per_graph, num_properties]

        # todo: add variance (reconstruct_field would return it)
        diff_properties = (input_properties - field_properties)   # [num_token_samples, batch, n_node_per_graph, num_properties]

        return field_properties, -0.5 * tf.reduce_mean(tf.reduce_sum(tf.math.square(diff_properties), axis=-1), axis=-1)   # [num_token_samples, batch, n_node_per_graph, num_properties], [num_token_samples, batch]

    def _build(self, graphs, imgs, **kwargs) -> dict:

        # graphs.nodes: [batch, n_node_per_graph, 3+num_properties]
        # imgs: [batch, H', W', C]
        latent_logits = self.encoder_2d(imgs) #batch, H, W, num_embeddings

        try:
            for variable in self.discrete_image_vae.encoder.trainable_variables:
                variable._trainable = False
        except:
            pass

        [_, H, W, num_embedding] = get_shape(latent_logits)

        latent_logits = tf.reshape(latent_logits, [self.batch, H*W, num_embedding])  # [batch, H*W, num_embeddings]
        reduce_logsumexp = tf.math.reduce_logsumexp(latent_logits, axis=-1)  # [batch, H*W]
        reduce_logsumexp = tf.tile(reduce_logsumexp[..., None], [1, 1, num_embedding]) # [batch, H*W, num_embedding]
        latent_logits -= reduce_logsumexp  # [batch, H*W, num_embeddings]

        temperature = tf.maximum(0.1, tf.cast(10. - 0.1 * (self.step / 1000), tf.float32))

        token_samples_onehot, token_samples = self.sample_latent_2d(latent_logits,
                                                                    temperature,
                                                                    self.num_token_samples) # [num_token_samples, batch, H*W, num_embedding / embedding_dim]

        # token_samples = tf.reshape(token_samples, [self.num_token_samples * self.batch, H * W, self.component_size])  # [num_token_samples*batch, H*W, embedding_dim]
        [_, _, n_node_per_graph, embedding_dim] = get_shape(token_samples)
        n_graphs = self.num_token_samples * self.batch

        token_samples = tf.reshape(token_samples, [n_graphs * n_node_per_graph, embedding_dim])  # [num_token_samples * batch * H * W, embedding_dim]

        token_graphs = GraphsTuple(nodes=token_samples,
                                   edges=None,
                                   globals=None,
                                   senders=None,
                                   receivers=None,
                                   n_node=tf.constant(n_graphs * [n_node_per_graph], dtype=tf.int32),
                                   n_edge=tf.constant(n_graphs * [0], dtype=tf.int32))  # [n_node, embedding_dim], n_node = n_graphs * n_node_per_graph

        tokens_3d, kl_div, token_3d_samples_onehot = self.decoder(token_graphs, temperature)  # [n_graphs, num_output, embedding_dim_3d], [n_graphs], [n_graphs, num_output, num_embedding_3d]
        tokens_3d = tf.reshape(tokens_3d, [self.num_token_samples,
                                           self.batch,
                                           self.num_components,
                                           self.component_size])  # [num_token_samples, batch, num_output, embedding_dim_3d]
        kl_div = tf.reshape(kl_div, [self.num_token_samples,
                                     self.batch])  # [num_token_samples, batch]

        # properties: [num_token_samples, batch, n_node_per_graph, 3+num_properties]
        # tokens_3d: [num_token_samples, batch, num_output, embedding_dim_3d]
        properties = tf.tile(graphs.nodes[None, ...], [self.num_token_samples, 1, 1, 1])
        field_properties, log_likelihood = self.log_likelihood(tokens_3d, properties)
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
            tf.summary.scalar('temperature', temperature, step=self.step)

            tf.summary.image('token_3d_samples_onehot', token_3d_samples_onehot[..., None], step=self.step)

        if self.step % 200 == 0:
            input_properties = graphs.nodes[0]
            reconstructed_properties = field_properties[0]
            pos = input_properties[:, :2]

            for i in range(num_channels):
                img_i = imgs[..., i][..., None]
                img_i = (img_i - tf.reduce_min(img_i)) / (
                        tf.reduce_max(img_i) - tf.reduce_min(img_i))
                tf.summary.image(f'img_before_autoencoder_{i}', img_i, step=self.step)

            for i in range(self.num_properties):
                image_before, _ = histogramdd(pos, bins=50, weights=input_properties[:, 3+i])
                image_before -= tf.reduce_min(image_before)
                image_before /= tf.reduce_max(image_before)
                tf.summary.image(f"{3+i}_xy_image_before_b", image_before[None, :, :, None], step=self.step)
                tf.summary.scalar(f"properties{3+i}_std_before", tf.math.reduce_std(input_properties[:, 3+i]), step=self.step)

                image_after, _ = histogramdd(pos, bins=50, weights=reconstructed_properties[:, i])
                image_after -= tf.reduce_min(image_after)
                image_after /= tf.reduce_max(image_after)
                tf.summary.image(f"{3+i}_xy_image_after", image_after[None, :, :, None], step=self.step)
                tf.summary.scalar(f"properties{3+i}_std_after", tf.math.reduce_std(reconstructed_properties[:, i]), step=self.step)

        return dict(loss=loss,
                    metrics=dict(var_exp=var_exp,
                                 kl_term=kl_div,
                                 mean_perplexity=mean_perplexity))

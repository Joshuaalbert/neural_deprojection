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
        
    def log_likelihood(self, tokens_3d, graphs):
        """
                Args:
                    tokens_3d: [num_token_samples, batch, n_tokens, embedding_dim_3d]

                    graphs: GraphsTuple
                        graph.nodes: [batch, n_node_per_graph, 3 + num_properties]

                Returns:
                    scalar
                """

        positions = graphs.nodes[:, :, :3]  # [batch, n_node_per_graph, 3]
        positions = tf.tile(positions[None, ...], [self.num_token_samples, 1, 1, 1])

        with tf.GradientTape() as tape:
            field_properties = self.reconstruct_field(tokens_3d, positions)  # [num_token_samples, batch, n_node_per_graph, num_properties]

        # field_properties = tf.reduce_mean(field_properties, axis=0)   # [batch, n_node_per_graph, num_properties]
        input_properties = graphs.nodes[:, :, 3:]  # [batch, n_node_per_graph, n_prop]
        input_properties = tf.tile(input_properties[None, ...], [self.num_token_samples, 1, 1, 1])   # [num_token_samples, batch, n_node_per_graph, num_properties]

        diff_properties = (input_properties - field_properties)   # [num_token_samples, batch, n_node_per_graph, num_properties]

        return field_properties, tf.reduce_mean(tf.reduce_sum(tf.math.square(diff_properties), axis=-1), axis=-1)   # [num_token_samples, batch, n_node_per_graph, num_properties], [num_token_samples, batch]

    # def kl_term(self, latent_logits, token_samples_onehot):
    #     # latent_logits [batch, n_node_per_graph, num_embeddings]
    #     # token_samples_onehot  [num_token_samples, batch, n_node_per_graph, num_embedding]
    #
    #     def _single_kl_term(token_sample_onehot):
    #         sum_selected_logits = tf.math.reduce_sum(token_sample_onehot * latent_logits, axis=-1)  # [batch, n_node_per_graph]
    #         sum_selected_logits = tf.reshape(sum_selected_logits, [batch, H, W])
    #         kl_term = tf.reduce_sum(sum_selected_logits, axis=[-2, -1])  # [batch]
    #         return kl_term
    #
    #     kl_term = tf.vectorized_map(_single_kl_term, token_samples_onehot)
    #     return kl_term

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

        temperature = tf.maximum(0.1, tf.cast(10. - 0.1 / (self.step / 1000), tf.float32))

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

        tokens_3d, kl_div = self.decoder(token_graphs, temperature)  # [n_graphs, num_output, embedding_dim_3d], [n_graphs]
        tokens_3d = tf.reshape(tokens_3d, [self.num_token_samples,
                                           self.batch,
                                           self.num_components,
                                           self.component_size])  # [num_token_samples, batch, num_output, embedding_dim_3d]
        kl_div = tf.reshape(kl_div, [self.num_token_samples,
                                     self.batch])  # [num_token_samples, batch]

        # graphs.nodes: [batch, n_node_per_graph, 3+num_properties]
        # tokens_3d: [num_token_samples, batch, num_output, embedding_dim_3d]
        field_properties, log_likelihood = self.log_likelihood(tokens_3d, graphs)
        # field_properties: [num_token_samples, batch, n_node_per_graph, num_properties]
        # log_likelihood: [num_token_samples, batch]

        field_properties = tf.reduce_mean(field_properties, axis=0)  # [batch, n_node_per_graph, num_properties]
        var_exp = tf.reduce_mean(log_likelihood, axis=0)  # [batch]
        kl_div = tf.reduce_mean(kl_div, axis=0)  # [batch]
        elbo = tf.reduce_mean(var_exp - kl_div)  # scalar

        entropy = -tf.reduce_sum(latent_logits * tf.math.exp(latent_logits), axis=-1)  # [batch, H, W]
        perplexity = 2. ** (-entropy / tf.math.log(2.))
        mean_perplexity = tf.reduce_mean(perplexity)

        loss = - elbo  # maximize ELBO so minimize -ELBO

        if self.step % 10 == 0:
            tf.summary.scalar('perplexity', mean_perplexity, step=self.step)
            tf.summary.scalar('var_exp', tf.reduce_mean(var_exp), step=self.step)
            tf.summary.scalar('kl_div', tf.reduce_mean(kl_div), step=self.step)
            tf.summary.scalar('temperature', temperature, step=self.step)

        if self.step % 200 == 0:
            input_properties = graphs.nodes[0]
            reconstructed_properties = field_properties[0]
            pos = input_properties[:, :2]

            imgs -= tf.reduce_min(imgs)
            imgs /= tf.reduce_max(imgs)
            tf.summary.image(f'img', imgs, step=self.step)

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

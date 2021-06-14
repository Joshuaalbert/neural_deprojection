from graph_nets import blocks
from graph_nets.utils_tf import concat

import tensorflow as tf
import sonnet as snt
from graph_nets.graphs import GraphsTuple
from graph_nets.utils_tf import fully_connect_graph_dynamic
from neural_deprojection.graph_net_utils import AbstractModule, gaussian_loss_function, get_shape
import tensorflow_probability as tfp


class SimpleCompleteModel(AbstractModule):
    def __init__(self,
                 num_properties: int,
                 discrete_image_vae,
                 graph_decoder,
                 num_token_samples: int,
                 name=None):
        super(SimpleCompleteModel, self).__init__(name=name)

        self.discrete_image_vae = discrete_image_vae
        self.decoder = graph_decoder
        self.num_token_samples = num_token_samples

        self.field_reconstruction = snt.nets.MLP([num_properties * 10 * 2, num_properties * 10 * 2, num_properties],
                                                 activate_final=False)

    def encoder_2d(self, img):
        latent_logits = self.discrete_image_vae.encoder(img)
        return latent_logits

    def sample_latent_2d(self, latent_logits):
        token_samples_onehot, token_samples = self.discrete_image_vae.sample_latent_2d(latent_logits,
                                                                                       self.num_token_samples)
        return token_samples_onehot, token_samples

    def reconstruct_field(self, field_component_tokens, positions):
        """
        Reconstruct the field at positions.

        Args:
            field_component_tokens: [batch, num_field_components, component_size]
            positions: [batch, n_node, 3]

        Returns:
            [batch, n_node, num_properties]
        """
        pos_shape = get_shape(positions)

        def _single_batch(args):
            """
            Compute for single batch.

            Args:
                tokens: [num_field_components, component_size]
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
                features = tf.concat([positions, tf.tile(token[None, :], [pos_shape[1], 1])], axis=-1)
                return self.field_reconstruction(features)

            return tf.reduce_sum(tf.vectorized_map(_single_component, tokens), axis=0)

        return tf.vectorized_map(_single_batch, (field_component_tokens, positions))

    def set_beta(self, beta):
        self.beta.assign(beta)

    def set_temperature(self, temperature):
        self.temperature.assign(temperature)

    def _build(self, graph, img, **kwargs) -> dict:

        latent_logits = self.encoder_2d(img) #batch, H, W, num_embeddings
        latent_logits -= tf.reduce_logsumexp(latent_logits, axis=-1, keepdims=True)
        token_samples_onehot, token_samples = self.sample_latent_2d(latent_logits)  # [num_token_samples, batch, H*W, embedding_dim]
        # todo: reshape [num_token_samples, batch, H*W, embedding_dim] -> [num_token_samples*batch, H*W, embedding_dim]
        likelihood_params = self.decoder(token_samples) # [num_token_samples*batch, num_reconstruction_components, reconstruction_comp_dim]
        #todo: unshape
        log_likelihood = self.log_likelihood(likelihood_params) #[num_token_samples, batch]
        kl_term = self.kl_term(latent_logits, token_samples_onehot) #[num_token_samples, batch]

        var_exp = tf.reduce_mean(log_likelihood, axis=0)#batch
        kl_div = tf.reduce_mean(kl_term, axis=0)#batch
        elbo = tf.reduce_mean(var_exp - kl_div)#scalar

        entropy = -tf.reduce_sum(latent_logits * tf.math.exp(latent_logits), axis=-1)  # batch, H, W
        perplexity = 2. ** (-entropy / tf.math.log(2.))
        mean_perplexity = tf.reduce_mean(perplexity)

        loss = - elbo  # maximize ELBO so minimize -ELBO

        tf.summary.scalar('perplexity', mean_perplexity, step=self.step)
        tf.summary.scalar('var_exp', var_exp, step=self.step)
        tf.summary.scalar('kl_term', kl_term, step=self.step)

        return dict(loss=loss,
                    metrics=dict(var_exp=var_exp,
                                 kl_term=kl_term,
                                 mean_perplexity=mean_perplexity))

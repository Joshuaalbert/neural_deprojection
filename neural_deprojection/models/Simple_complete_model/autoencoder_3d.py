import tensorflow as tf
from neural_deprojection.graph_net_utils import AbstractModule, get_shape, grid_graphs, map_coordinates
import tensorflow_probability as tfp
from neural_deprojection.models.openai_dvae_modules.modules import Encoder3D, Decoder3D
from graph_nets.graphs import GraphsTuple


class DiscreteVoxelsVAE(AbstractModule):
    def __init__(self,
                 hidden_size: int = 8,
                 embedding_dim: int = 64,
                 num_embedding: int = 1024,
                 num_token_samples: int = 4,
                 num_channels:int = 1,
                 voxels_per_dimension:int = 64,
                 compute_temperature:callable = None,
                 beta: float = 1.,
                 num_groups=4,
                 name=None):
        super(DiscreteVoxelsVAE, self).__init__(name=name)
        self.voxels_per_dimension = voxels_per_dimension
        self.num_channels = num_channels
        self.embeddings = tf.Variable(initial_value=tf.random.truncated_normal((num_embedding, embedding_dim)),
                                      name='embeddings')
        self.num_token_samples = num_token_samples
        self.num_embedding = num_embedding
        self.embedding_dim = embedding_dim
        self.compute_temperature = compute_temperature
        self.beta = tf.convert_to_tensor(beta, dtype=tf.float32)

        self._encoder = Encoder3D(hidden_size=hidden_size, num_embeddings=num_embedding, num_groups=num_groups,name='EncoderVoxels')
        self._decoder = Decoder3D(hidden_size=hidden_size, num_channels=num_channels, num_groups=num_groups,name='DecoderVoxels')

        assert self._encoder.shrink_factor == self._decoder.shrink_factor, "Shrink factors should be same. Use same num_groups."
        self.shrink_factor = self._decoder.shrink_factor

    @property
    def temperature(self):
        return self.compute_temperature(self.epoch)

    def set_beta(self, beta):
        self.beta.assign(beta)

    def set_temperature(self, temperature):
        self.temperature.assign(temperature)

    def compute_logits(self, graphs):
        """
        Computes normalised logits representing the variational posterior, q(z | img).

        Args:
            graphs: GraphsTuple

        Returns:
            logits: [batch, W, H, D, num_embeddings]
        """
        #[batch, H', W', D', num_channels]
        img = grid_graphs(graphs, voxels_per_dimension=self.voxels_per_dimension)
        # number of channels must be known for conv3d
        img.set_shape([None, None, None, None, self.num_channels])
        logits = self._encoder(img)
        logits /= 1e-6 + tf.math.reduce_std(logits, axis=-1, keepdims=True)
        logits -= tf.reduce_logsumexp(logits, axis=-1, keepdims=True)
        return logits

    def sample_latent(self, logits, temperature, num_samples):
        """
        Sample one-hot encodings of each latent variable.

        Args:
            logits: [batch, W, H, D, num_embeddings]
            num_samples: int

        Returns:
            log_token_samples_onehot, token_samples_onehot: [num_samples, batch, W, H, D, num_embeddings]
            latent_tokens: [num_samples, batch, W, H, D, embedding_size]
        """
        token_distribution = tfp.distributions.ExpRelaxedOneHotCategorical(temperature, logits=logits)
        log_token_samples_onehot = token_distribution.sample((num_samples,), name='token_samples')  # [S, batch, W, H, D, num_embeddings]
        token_samples_onehot = tf.math.exp(log_token_samples_onehot)
        latent_tokens = tf.einsum("sbwhdn,nm->sbwhdm",token_samples_onehot, self.embeddings) # [S, batch, W, H, D, embedding_size]
        return log_token_samples_onehot, token_samples_onehot, latent_tokens

    def compute_likelihood_parameters(self, latent_tokens):
        """
        Compute the likelihood parameters from logits.

        Args:
            latent_tokens: [num_samples, batch, W, H, D, embedding_size]

        Returns:
            mu, logb: [num_samples, batch, W', H', D', num_channels]

        """
        [num_samples, batch, H, W, D, _] = get_shape(latent_tokens)
        latent_tokens = tf.reshape(latent_tokens, [num_samples*batch, H, W, D, self.embedding_dim])  #[num_samples*batch, H, W, D, self.embedding_dim]
        decoded_imgs = self._decoder(latent_tokens)  # [num_samples * batch, H', W', D', C*2]
        decoded_imgs.set_shape([None,
                                self.voxels_per_dimension, self.voxels_per_dimension, self.voxels_per_dimension,
                                self.num_channels*2])
        [_, H_2, W_2, D_2, _] = get_shape(decoded_imgs)
        decoded_imgs = tf.reshape(decoded_imgs, [num_samples, batch, H_2, W_2, D_2, 2 * self.num_channels])  # [S, batch, H, W, D, embedding_dim]
        mu, logb = decoded_imgs[..., :self.num_channels], decoded_imgs[..., self.num_channels:]
        return mu, logb  # [S, batch, H', W', D' C], [S, batch, H', W', D' C]

    def log_likelihood(self, graphs: GraphsTuple, mu, logb):
        """
        Log-Laplace distribution.

        The pdf of log-Laplace is,

            P(x | mu, b) = 1 / (2 * log(x) * b) * exp(|log(x) - mu|/b)

        Args:
            graphs: GraphsTuple in standard form. Assumes properties are of the form log(maximum(1e-5, properties))
            mu: [num_samples, batch, H', W', D', channels]
            logb: [num_samples, batch, H', W', D', channels]

        Returns:
            log_prob [num_samples, batch]
        """
        properties = grid_graphs(graphs, self.voxels_per_dimension) # [num_samples, batch, H', W', D', num_properties]
        log_prob = - tf.math.abs(properties - mu) / tf.math.exp(logb) \
                   - tf.math.log(2.) - properties - logb # [num_samples, batch, H', W', D', num_properties]
        #num_samples, batch
        return tf.reduce_sum(log_prob, axis=[-1, -2, -3, -4])

    def log_prob_q(self,latent_logits, log_token_samples_onehot):
        """
        Args:
            latent_logits: [batch, H, W, D, num_embeddings] (normalised)
            log_token_samples_onehot: [num_samples, batch, H, W, D, num_embeddings]

        Returns:

        """
        _, H, W, D, _ = get_shape(latent_logits)
        q_dist = tfp.distributions.ExpRelaxedOneHotCategorical(self.temperature, logits=latent_logits)
        log_prob_q = q_dist.log_prob(log_token_samples_onehot)  # num_samples, batch, H, W, D
        return log_prob_q

    def kl_term(self, latent_logits, log_token_samples_onehot):
        """
        Compute the term, which if marginalised over q(z) results in KL(q | prior).

            sum_z log q(z) - log prior(z)

        Args:
            latent_logits: [batch, H, W, D, num_embeddings] (normalised)
            token_samples_onehot: [num_samples, batch, H, W, D, num_embeddings]

        Returns:
            kl_term: [num_samples, batch]
        """
        log_prob_q = self.log_prob_q(latent_logits, log_token_samples_onehot)  # num_samples, batch, H, W, D

        prior_dist = tfp.distributions.ExpRelaxedOneHotCategorical(self.temperature, logits=tf.zeros_like(latent_logits))
        log_prob_prior = prior_dist.log_prob(log_token_samples_onehot)  # num_samples, batch, H, W, D
        return tf.reduce_sum(log_prob_q - log_prob_prior, axis=[-1, -2, -3])  # num_samples, batch

    def _build(self, graphs, **kwargs) -> dict:
        """
        Args:
            graphs: GraphsTuple in standard form.
            **kwargs:

        Returns:

        """
        latent_logits = self.compute_logits(graphs) # [batch, H, W, D, num_embeddings]
        log_token_samples_onehot, token_samples_onehot, latent_tokens = self.sample_latent(latent_logits, self.temperature, self.num_token_samples) # [num_samples, batch, H, W, D, num_embeddings], [num_samples, batch, H, W, D, embedding_size]
        mu, logb = self.compute_likelihood_parameters(latent_tokens) # [num_samples, batch, H', W', D', C], [num_samples, batch, H', W', D', C]
        log_likelihood = self.log_likelihood(graphs, mu, logb) # [num_samples, batch]
        kl_term = self.kl_term(latent_logits, log_token_samples_onehot) # [num_samples, batch]

        var_exp = tf.reduce_mean(log_likelihood, axis=0)  # [batch]
        kl_div = tf.reduce_mean(kl_term, axis=0)  # [batch]
        elbo = var_exp - self.beta * kl_div  # batch
        loss = - tf.reduce_mean(elbo)  # scalar

        entropy = -tf.reduce_sum(tf.math.exp(latent_logits) * latent_logits, axis=[-1])  # [batch, H, W, D]
        perplexity = 2. ** (entropy / tf.math.log(2.))  # [batch, H, W, D]
        mean_perplexity = tf.reduce_mean(perplexity)  # scalar

        if self.log_counter % int(128 / mu.get_shape()[0]) == 0:
            tf.summary.scalar('perplexity', mean_perplexity, step=self.step)
            tf.summary.scalar('var_exp', tf.reduce_mean(var_exp), step=self.step)
            tf.summary.scalar('kl_div', tf.reduce_mean(kl_div), step=self.step)
            tf.summary.scalar('temperature', self.temperature, step=self.step)
            tf.summary.scalar('beta', self.beta, step=self.step)

            projected_mu = tf.reduce_sum(mu[0], axis=-2) #[batch, H', W', C]
            voxels = grid_graphs(graphs, self.voxels_per_dimension) #[batch, H', W', D', C]
            projected_img = tf.reduce_sum(voxels, axis=-2) #[batch, H', W', C]
            for i in range(self.num_channels):
                vmin = tf.reduce_min(projected_mu[..., i])
                vmax = tf.reduce_max(projected_mu[..., i])
                _projected_mu = (projected_mu[..., i:i+1]-vmin)/(vmax-vmin)#batch, H', W', 1
                _projected_mu = tf.clip_by_value(_projected_mu, 0., 1.)
                vmin = tf.reduce_min(projected_img[..., i])
                vmax = tf.reduce_max(projected_img[..., i])
                _projected_img = (projected_img[..., i:i+1]-vmin)/(vmax-vmin)#batch, H', W', 1
                _projected_img = tf.clip_by_value(_projected_img, 0., 1.)

                tf.summary.image(f'voxels_predict[{i}]', _projected_mu, step=self.step)
                tf.summary.image(f'voxels_actual[{i}]', _projected_img, step=self.step)


            batch, H, W, D, _ = get_shape(latent_logits)
            _latent_logits = latent_logits # [batch, H, W, D, num_embeddings]
            _latent_logits -= tf.reduce_min(_latent_logits, axis=-1, keepdims=True)
            _latent_logits /= tf.reduce_max(_latent_logits, axis=-1, keepdims=True)
            _latent_logits = tf.reshape(_latent_logits, [batch, H*W*D, self.num_embedding, 1])  # [batch, H*W*D, num_embedding, 1]
            tf.summary.image('latent_logits', _latent_logits, step=self.step)

            token_sample_onehot = token_samples_onehot[0] # [batch, H, W, D, num_embeddings]
            token_sample_onehot = tf.reshape(token_sample_onehot, [batch, H*W*D, self.num_embedding, 1])  # [batch, H*W*D, num_embedding, 1]
            tf.summary.image('latent_samples_onehot', token_sample_onehot, step=self.step)

        return dict(loss=loss,
                    metrics=dict(var_exp=var_exp,
                                 kl_div=kl_div,
                                 mean_perplexity=mean_perplexity))

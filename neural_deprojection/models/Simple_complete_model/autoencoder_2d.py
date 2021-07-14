import tensorflow as tf
from neural_deprojection.graph_net_utils import AbstractModule, get_shape
import tensorflow_probability as tfp
from neural_deprojection.models.openai_dvae_modules.modules import Encoder2D, Decoder2D

class DiscreteImageVAE(AbstractModule):
    def __init__(self,
                 hidden_size: int = 32,
                 embedding_dim: int = 64,
                 num_embedding: int = 1024,
                 num_token_samples: int = 4,
                 num_channels:int = 1,
                 temperature:float = 1.,
                 beta:float = 1.,
                 name=None):
        super(DiscreteImageVAE, self).__init__(name=name)
        self.num_channels = num_channels
        self.embeddings = tf.Variable(initial_value=tf.random.truncated_normal((num_embedding, embedding_dim)),
                                      name='embeddings')
        self.num_token_samples = num_token_samples
        self.num_embedding = num_embedding
        self.embedding_dim = embedding_dim
        self.temperature = tf.convert_to_tensor(temperature)
        self.beta = tf.convert_to_tensor(beta)

        self._encoder = Encoder2D(hidden_size=hidden_size, num_embeddings=num_embedding, name='EncoderImage')
        self._decoder = Decoder2D(hidden_size=hidden_size, num_channels=num_channels, name='DecoderImage')

    def set_beta(self, beta):
        self.beta.assign(beta)

    def set_temperature(self, temperature):
        self.temperature.assign(temperature)

    def compute_logits(self, img):
        """
        Computes normalised logits representing the variational posterior, q(z | img).

        Args:
            img: [batch, W', H', num_channels]

        Returns:
            logits: [batch, W, H, num_embeddings]
        """
        #[batch, W, H, num_embeddings]
        logits = self._encoder(img)
        logits /= 1e-5 + tf.math.reduce_std(logits, axis=-1, keepdims=True)
        logits -= tf.reduce_logsumexp(logits, axis=-1, keepdims=True)

        return logits

    def sample_latent(self, logits, temperature, num_samples):
        """
        Sample one-hot encodings of each latent variable.

        Args:
            logits: [batch, W, H, num_embeddings]
            num_samples: int

        Returns:
            token_samples_onehot: [num_samples, batch, W, H, num_embeddings]
            latent_tokens: [num_samples, batch, W, H, embedding_size]
        """
        token_distribution = tfp.distributions.RelaxedOneHotCategorical(temperature, logits=logits)
        token_samples_onehot = token_distribution.sample((num_samples,), name='token_samples')  # [S, batch, W, H, num_embeddings]
        latent_tokens = tf.einsum("sbwhn,nm->sbwhm",token_samples_onehot, self.embeddings) # [S, batch, W, H, embedding_size]
        return token_samples_onehot, latent_tokens

    def compute_likelihood_parameters(self, latent_tokens):
        """
        Compute the likelihood parameters from logits.

        Args:
            latent_tokens: [num_samples, batch, W, H, embedding_size]

        Returns:
            mu, logb: [num_samples, batch, W', H', num_channels]

        """
        [num_samples, batch, H, W, _] = get_shape(latent_tokens)
        latent_tokens = tf.reshape(latent_tokens, [num_samples*batch, H, W, self.embedding_dim])  #[num_samples*batch, H, W, self.embedding_dim]
        decoded_imgs = self._decoder(latent_tokens)  # [num_samples * batch, H', W', C*2]
        decoded_imgs.set_shape([None,
                                None, None,
                                self.num_channels*2])
        [_, H_2, W_2, _] = get_shape(decoded_imgs)
        decoded_imgs = tf.reshape(decoded_imgs, [num_samples, batch, H_2, W_2, 2 * self.num_channels])  # [S, batch, H, W, embedding_dim]
        mu, logb = decoded_imgs[..., :self.num_channels], decoded_imgs[..., self.num_channels:]
        return mu, logb  # [S, batch, H', W' C], [S, batch, H', W' C]

    def log_likelihood(self, properties, mu, logb):
        """
        Log-Laplace distribution.

        The pdf of log-Laplace is,

            P(x | mu, b) = 1 / (2 * log(x) * b) * exp(|log(x) - mu|/b)

        Args:
            properties: image data [batch, H', W', channels]. Assumes properties are of the form log(maximum(1e-5, properties))
            mu: [num_samples, batch, H', W', channels]
            logb: [num_samples, batch, H', W', channels]

        Returns:
            log_prob [num_samples, batch]
        """

        log_prob = - tf.math.abs(properties - mu) / tf.math.exp(logb) \
                   - tf.math.log(2.) - properties - logb # [num_samples, batch, H', W', num_properties]
        #num_samples, batch
        return tf.reduce_sum(log_prob, axis=[-1, -2, -3])

    def kl_term(self, latent_logits, token_samples_onehot):
        """
        Compute the term, which if marginalised over q(z) results in KL(q | prior).

            sum_z log q(z) - log prior(z)

        Args:
            latent_logits: [batch, H, W, num_embeddings] (normalised)
            token_samples_onehot: [num_samples, batch, H, W, num_embeddings]

        Returns:
            kl_term: [num_samples, batch]
        """
        _, H, W, _ = get_shape(latent_logits)
        q_dist = tfp.distributions.RelaxedOneHotCategorical(self.temperature, logits=latent_logits)
        log_prob_q = q_dist.log_prob(token_samples_onehot) #num_samples, batch, H, W

        prior_dist = tfp.distributions.RelaxedOneHotCategorical(self.temperature, logits=tf.zeros_like(latent_logits))
        log_prob_prior = prior_dist.log_prob(token_samples_onehot) # num_samples, batch, H, W
        return tf.reduce_sum(log_prob_q - log_prob_prior, axis=[-1,-2])# num_samples, batch

    def _build(self, img, **kwargs) -> dict:
        """
        Args:
            img: [batch, H, W, num_channels]
            **kwargs:

        Returns:

        """
        latent_logits = self.compute_logits(img) # [batch, H, W, num_embeddings]
        token_samples_onehot, latent_tokens = self.sample_latent(latent_logits, self.temperature, self.num_token_samples) # [num_samples, batch, H, W, num_embeddings], [num_samples, batch, H, W, embedding_size]
        mu, logb = self.compute_likelihood_parameters(latent_tokens) # [num_samples, batch, H', W', C], [num_samples, batch, H', W', C]
        log_likelihood = self.log_likelihood(img, mu, logb) # [num_samples, batch]
        kl_term = self.kl_term(latent_logits, token_samples_onehot) # [num_samples, batch]

        var_exp = tf.reduce_mean(log_likelihood, axis=0)  # [batch]
        kl_div = tf.reduce_mean(kl_term, axis=0)  # [batch]
        elbo = var_exp - self.beta * kl_div  # batch
        loss = - tf.reduce_mean(elbo)  # scalar

        entropy = -tf.reduce_sum(latent_logits * token_samples_onehot, axis=-1)  # [S, batch, H, W]
        perplexity = 2. ** (entropy / tf.math.log(2.)) # [S, batch, H, W]
        mean_perplexity = tf.reduce_mean(perplexity)  # scalar

        if self.step % 10 == 0:
            tf.summary.scalar('perplexity', mean_perplexity, step=self.step)
            tf.summary.scalar('var_exp', tf.reduce_mean(var_exp), step=self.step)
            tf.summary.scalar('kl_div', tf.reduce_mean(kl_div), step=self.step)
            tf.summary.scalar('temperature', self.temperature, step=self.step)
            tf.summary.scalar('beta', self.beta, step=self.step)

            _mu = mu[0] #[batch, H', W', C]
            _img = img #[batch, H', W', C]
            for i in range(self.num_channels):
                vmin = tf.reduce_min(_mu[..., i])
                vmax = tf.reduce_max(_mu[..., i])
                _projected_mu = (_mu[..., i:i+1]-vmin)/(vmax-vmin)#batch, H', W', 1
                _projected_mu = tf.clip_by_value(_projected_mu, 0., 1.)
                vmin = tf.reduce_min(_img[..., i])
                vmax = tf.reduce_max(_img[..., i])
                _projected_img = (_img[..., i:i+1]-vmin)/(vmax-vmin)#batch, H', W', 1
                _projected_img = tf.clip_by_value(_projected_img, 0., 1.)

                tf.summary.image(f'image_predict[{i}]', _projected_mu, step=self.step)
                tf.summary.image(f'image_actual[{i}]', _projected_img, step=self.step)


            batch, H, W, _ = get_shape(latent_logits)
            _latent_logits = latent_logits # [batch, H, W, num_embeddings]
            _latent_logits -= tf.reduce_min(_latent_logits, axis=-1, keepdims=True)
            _latent_logits /= tf.reduce_max(_latent_logits, axis=-1, keepdims=True)
            _latent_logits = tf.reshape(_latent_logits, [batch, H*W, self.num_embedding, 1])  # [batch, H*W, num_embedding, 1]
            tf.summary.image('latent_logits', _latent_logits, step=self.step)

            token_sample_onehot = token_samples_onehot[0] # [batch, H, W, num_embeddings]
            token_sample_onehot = tf.reshape(token_sample_onehot, [batch, H*W, self.num_embedding, 1])  # [batch, H*W, num_embedding, 1]
            tf.summary.image('latent_samples_onehot', token_sample_onehot, step=self.step)

        return dict(loss=loss,
                    metrics=dict(var_exp=var_exp,
                                 kl_div=kl_div,
                                 mean_perplexity=mean_perplexity))

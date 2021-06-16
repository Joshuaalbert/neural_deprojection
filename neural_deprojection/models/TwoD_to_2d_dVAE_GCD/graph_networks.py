import sys

sys.path.insert(1, '/data/s2675544/git/neural_deprojection/')
sys.path.insert(1, '/home/matthijs/git/neural_deprojection/')

from graph_nets import blocks
from graph_nets.utils_tf import concat

import tensorflow as tf
import sonnet as snt
from graph_nets.graphs import GraphsTuple
from graph_nets.utils_tf import fully_connect_graph_dynamic, fully_connect_graph_static
from neural_deprojection.graph_net_utils import AbstractModule, histogramdd, get_shape
import tensorflow_probability as tfp
from neural_deprojection.models.openai_dvae_modules.modules import Encoder, Decoder

class DiscreteImageVAE(AbstractModule):
    def __init__(self,
                 hidden_size: int = 64,
                 embedding_dim: int = 64,
                 num_embedding: int = 1024,
                 num_token_samples: int = 32,
                 num_channels=1,
                 name=None):
        super(DiscreteImageVAE, self).__init__(name=name)
        # (num_embedding, embedding_dim)
        self.num_channels=num_channels
        self.embeddings = tf.Variable(initial_value=tf.random.truncated_normal((num_embedding, embedding_dim)),
                                      name='embeddings')
        self.num_token_samples = num_token_samples
        self.num_embedding = num_embedding
        self.embedding_dim = embedding_dim
        self.temperature = tf.Variable(initial_value=tf.constant(1.), name='temperature', trainable=False)
        self.beta = tf.Variable(initial_value=tf.constant(6.6), name='beta', trainable=False)

        self.encoder = Encoder(hidden_size=hidden_size, num_embeddings=num_embedding, name='EncoderImage')
        self.decoder = Decoder(hidden_size=hidden_size, num_channels=num_channels, name='DecoderImage')

    def set_beta(self, beta):
        self.beta.assign(beta)

    def set_temperature(self, temperature):
        self.temperature.assign(temperature)

    @tf.function(input_signature=[tf.TensorSpec([None, None, None, None], dtype=tf.float32)])
    def sample_encoder(self, img):
        return self.encoder(img)

    @tf.function(input_signature=[tf.TensorSpec([None, None, None, None], dtype=tf.float32),
                                  tf.TensorSpec([], dtype=tf.float32),
                                  tf.TensorSpec([], dtype=tf.float32)])
    def sample_decoder(self, img_logits, temperature, num_samples):
        [batch, H, W, _] = get_shape(img_logits)

        logits = tf.reshape(img_logits, [batch * H * W, self.num_embedding])  # [batch*H*W, num_embeddings]
        reduce_logsumexp = tf.math.reduce_logsumexp(logits, axis=-1)  # [batch*H*W]
        reduce_logsumexp = tf.tile(reduce_logsumexp[:, None], [1, self.num_embedding])  # [batch*H*W, num_embedding]
        logits -= reduce_logsumexp  # [batch*H*W, num_embeddings]
        token_distribution = tfp.distributions.RelaxedOneHotCategorical(temperature, logits=logits)
        token_samples_onehot = token_distribution.sample((num_samples,),
                                                         name='token_samples')  # [S, batch*H*W, num_embeddings]
        def _single_decode(token_sample_onehot):
            # [batch*H*W, num_embeddings] @ [num_embeddings, embedding_dim]
            token_sample = tf.matmul(token_samples_onehot[0], self.embeddings)  # [batch*H*W, embedding_dim]  # = z ~ q(z|x)
            latent_img = tf.reshape(token_sample, [batch, H, W, self.embedding_dim])  # [batch, H, W, embedding_dim]
            decoded_img = self.decoder(latent_img)  # [batch, H', W', C*2]
            return decoded_img

        decoded_ims = tf.vectorized_map(_single_decode, token_samples_onehot)  # [S, batch, H', W', C*2]
        decoded_im = tf.reduce_mean(decoded_ims, axis=0)  # [batch, H', W', C*2]
        return decoded_im


    def log_likelihood(self, img, mu, logb):
        """
        Log-Laplace distribution.

        Args:
            img: [...,c] assumes of the form log(maximum(1e-5, img))
            mu: [...,c]
            logb: [...,c]

        Returns:
            log_prob [...]
        """
        log_prob = - tf.math.abs(img - mu) / tf.math.exp(logb) \
                   - tf.math.log(2.) - img - logb
        return tf.reduce_sum(log_prob, axis=-1)


    def _build(self, img, **kwargs) -> dict:
        """

        Args:
            img: [batch, H', W', num_channel]
            **kwargs:

        Returns:

        """
        encoded_img_logits = self.encoder(img)  # [batch, H, W, num_embedding]
        [batch, H, W, _] = get_shape(encoded_img_logits)

        logits = tf.reshape(encoded_img_logits, [batch*H*W, self.num_embedding])  # [batch*H*W, num_embeddings]
        reduce_logsumexp = tf.math.reduce_logsumexp(logits, axis=-1)  # [batch*H*W]
        reduce_logsumexp = tf.tile(reduce_logsumexp[:, None], [1, self.num_embedding])  # [batch*H*W, num_embedding]
        logits -= reduce_logsumexp  # [batch*H*W, num_embeddings]

        temperature = tf.maximum(0.1, tf.cast(1. - 0.1/(self.step/1000), tf.float32))
        token_distribution = tfp.distributions.RelaxedOneHotCategorical(temperature, logits=logits)
        token_samples_onehot = token_distribution.sample((self.num_token_samples,), name='token_samples')  # [S, batch*H*W, num_embeddings]

        def _single_decode(token_sample_onehot):
            #[batch*H*W, num_embeddings] @ [num_embeddings, embedding_dim]
            token_sample = tf.matmul(token_sample_onehot, self.embeddings)  # [batch*H*W, embedding_dim]  # = z ~ q(z|x)
            latent_img = tf.reshape(token_sample, [batch, H, W, self.embedding_dim])  # [batch, H, W, embedding_dim]
            decoded_img = self.decoder(latent_img)  # [batch, H', W', C*2]
            # print('decod shape', decoded_img)
            img_mu = decoded_img[..., :self.num_channels] #[batch, H', W', C]
            # print('mu shape', img_mu)
            img_logb = decoded_img[..., self.num_channels:]
            # print('logb shape', img_logb)
            log_likelihood = self.log_likelihood(img, img_mu, img_logb)#[batch, H', W', C]
            log_likelihood = tf.reduce_sum(log_likelihood, axis=[-3,-2,-1])  # [batch]
            sum_selected_logits = tf.math.reduce_sum(token_sample_onehot * logits, axis=-1)  # [batch*H*W]
            sum_selected_logits = tf.reshape(sum_selected_logits, [batch, H, W])
            kl_term = tf.reduce_sum(sum_selected_logits, axis=[-2,-1])#[batch]
            return log_likelihood, kl_term, decoded_img

        #num_samples, batch
        log_likelihood_samples, kl_term_samples, decoded_ims = tf.vectorized_map(_single_decode, token_samples_onehot)  # [S, batch], [S, batch]

        if self.step % 50 == 0:
            img_mu_0 = tf.reduce_mean(decoded_ims, axis=0)[..., :self.num_channels]
            img_mu_0 -= tf.reduce_min(img_mu_0)
            img_mu_0 /= tf.reduce_max(img_mu_0)
            tf.summary.image('mu', img_mu_0, step=self.step)

            smoothed_img = img[..., self.num_channels:]
            smoothed_img = (smoothed_img - tf.reduce_min(smoothed_img)) / (
                    tf.reduce_max(smoothed_img) - tf.reduce_min(smoothed_img))
            tf.summary.image(f'img_before_autoencoder', smoothed_img, step=self.step)

        var_exp = tf.reduce_mean(log_likelihood_samples, axis=0)  # [batch]
        kl_div = tf.reduce_mean(kl_term_samples, axis=0)  # [batch]
        elbo = var_exp - kl_div  # batch
        loss = - tf.reduce_mean(elbo)  # scalar

        entropy = -tf.reduce_sum(logits * tf.math.exp(logits), axis=-1)  # [batch*H*W]
        perplexity = 2. ** (-entropy / tf.math.log(2.))  # [batch*H*W]
        mean_perplexity = tf.reduce_mean(perplexity)  # scalar

        if self.step % 2 == 0:
            logits = tf.nn.softmax(logits, axis=-1)  # [batch*H*W, num_embedding]
            logits -= tf.reduce_min(logits)
            logits /= tf.reduce_max(logits)
            logits = tf.reshape(logits, [batch, H*W, self.num_embedding])[0]  # [H*W, num_embedding]
            # tf.repeat(tf.repeat(logits, 16*[4], axis=0), 512*[4], axis=1)
            tf.summary.image('logits', logits[None, :, :, None], step=self.step)
            tf.summary.scalar('perplexity', mean_perplexity, step=self.step)
            tf.summary.scalar('var_exp', tf.reduce_mean(var_exp), step=self.step)
            tf.summary.scalar('kl_div', tf.reduce_mean(kl_div), step=self.step)


        return dict(loss=loss,
                    metrics=dict(var_exp=var_exp,
                                 kl_div=kl_div,
                                 mean_perplexity=mean_perplexity))

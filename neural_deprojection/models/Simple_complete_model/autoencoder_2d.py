import tensorflow as tf
from neural_deprojection.graph_net_utils import AbstractModule, histogramdd, get_shape
import tensorflow_probability as tfp
from neural_deprojection.models.openai_dvae_modules.modules import Encoder, Decoder
import sonnet as snt


class ResidualStack(AbstractModule):
    def __init__(self,
                 num_hiddens,
                 num_residual_layers,
                 num_residual_hiddens,
                 residual_name='',
                 name=None):
        super(ResidualStack, self).__init__(name=name)
        self._num_hiddens = num_hiddens
        self._num_residual_layers = num_residual_layers
        self._num_residual_hiddens = num_residual_hiddens

        self._layers = []
        for i in range(num_residual_layers):
            conv3 = snt.Conv2D(
                output_channels=num_residual_hiddens,
                kernel_shape=(3, 3),
                stride=(1, 1),
                name=f"res3x3_{residual_name}_{i}")
            conv1 = snt.Conv2D(
                output_channels=num_hiddens,
                kernel_shape=(1, 1),
                stride=(1, 1),
                name=f"res1x1_{residual_name}_{i}")
            self._layers.append((conv3, conv1))

    def _build(self, inputs):
        h = inputs
        for conv3, conv1 in self._layers:
            conv3_out = conv3(tf.nn.relu(h))
            conv1_out = conv1(tf.nn.relu(conv3_out))
            h += conv1_out
        return tf.nn.relu(h)  # Resnet V1 style


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
        def _single_latent_img(token_sample_onehot):
            # [batch*H*W, num_embeddings] @ [num_embeddings, embedding_dim]
            token_sample = tf.matmul(token_sample_onehot, self.embeddings)  # [batch*H*W, embedding_dim]  # = z ~ q(z|x)
            latent_img = tf.reshape(token_sample, [batch, H, W, self.embedding_dim])  # [batch, H, W, embedding_dim]
            return latent_img

        latent_imgs = tf.vectorized_map(_single_latent_img, token_samples_onehot)  # [S, batch, H, W, embedding_dim]
        latent_imgs = tf.reshape(latent_imgs, [self.num_token_samples * batch, H, W, self.embedding_dim])  # [S * batch, H, W, embedding_dim]
        decoded_imgs = self.decoder(latent_imgs)  # [S * batch, H', W', C*2]
        [_, H_2, W_2, _] = get_shape(decoded_imgs)
        decoded_imgs = tf.reshape(decoded_imgs, [self.num_token_samples, batch, H_2, W_2, 2 * self.num_channels])  # [S, batch, H, W, embedding_dim]
        decoded_img = tf.reduce_mean(decoded_imgs, axis=0)  # [batch, H', W', C*2]
        return decoded_img

    @tf.function(input_signature=[tf.TensorSpec([None, None, None], dtype=tf.float32),
                                  tf.TensorSpec([], dtype=tf.float32),
                                  tf.TensorSpec([], dtype=tf.float32)])
    def sample_latent_2d(self, latent_logits, temperature, num_token_samples):
        return self._sample_latent_2d(latent_logits, temperature, num_token_samples)

    def _sample_latent_2d(self, latent_logits, temperature, num_token_samples):
        token_distribution = tfp.distributions.RelaxedOneHotCategorical(temperature, logits=latent_logits)
        token_samples_onehot = token_distribution.sample((num_token_samples,),
                                                         name='token_samples')  # [S, batch, H*W, num_embeddings]

        def _single_decode(token_sample_onehot):
            # [batch, H*W, num_embeddings] @ [num_embeddings, embedding_dim]
            token_sample = tf.matmul(token_sample_onehot,
                                     self.embeddings)  # [batch, H*W, embedding_dim]  # = z ~ q(z|x)
            return token_sample

        token_samples = tf.vectorized_map(_single_decode, token_samples_onehot)  # [S, batch, H*W, embedding_dim]
        return token_samples_onehot, token_samples


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

        # temperature = tf.maximum(0.1, tf.cast(10. - 0.1 * (self.step / 1000), tf.float32))
        token_distribution = tfp.distributions.RelaxedOneHotCategorical(self.temperature, logits=logits)
        # token_distribution = tfp.distributions.OneHotCategorical(logits=logits)
        token_samples_onehot = token_distribution.sample((self.num_token_samples,), name='token_samples')  # [S, batch*H*W, num_embeddings]
        token_samples_onehot = tf.cast(token_samples_onehot, dtype=tf.float32)

        def _single_decode_part_1(token_sample_onehot):
            #[batch*H*W, num_embeddings] @ [num_embeddings, embedding_dim]
            token_sample = tf.matmul(token_sample_onehot, self.embeddings)  # [batch*H*W, embedding_dim]  # = z ~ q(z|x)
            latent_img = tf.reshape(token_sample, [batch, H, W, self.embedding_dim])  # [batch, H, W, embedding_dim]
            return latent_img

        def _single_decode_part_2(args):
            decoded_img, token_sample_onehot = args
            img_mu = decoded_img[..., :self.num_channels] #[batch, H', W', C]
            img_logb = decoded_img[..., self.num_channels:]
            log_likelihood = self.log_likelihood(img, img_mu, img_logb)#[batch, H', W']
            log_likelihood = tf.reduce_sum(log_likelihood, axis=[-2,-1])  # [batch]
            sum_selected_logits = tf.math.reduce_sum(token_sample_onehot * logits, axis=-1)  # [batch*H*W]
            sum_selected_logits = tf.reshape(sum_selected_logits, [batch, H, W])
            kl_term = tf.reduce_sum(sum_selected_logits, axis=[-2,-1])#[batch]
            return log_likelihood, kl_term, decoded_img

        latent_imgs = tf.vectorized_map(_single_decode_part_1, token_samples_onehot)  # [S, batch, H, W, embedding_dim]

        # decoder outside the vectorized map, first merge batch and sample dimension
        latent_imgs = tf.reshape(latent_imgs, [self.num_token_samples * batch, H, W, self.embedding_dim])  # [S * batch, H, W, embedding_dim]
        decoded_imgs = self.decoder(latent_imgs)  # [S * batch, H', W', C*2]

        # reshape again for input of the second vectorized map
        [_, H_2, W_2, _] = get_shape(decoded_imgs)
        decoded_imgs = tf.reshape(decoded_imgs, [self.num_token_samples, batch, H_2, W_2, 2 * self.num_channels])  # [S, batch, H, W, embedding_dim]
        log_likelihood_samples, kl_term_samples, decoded_ims = tf.vectorized_map(_single_decode_part_2, (decoded_imgs, token_samples_onehot))

        var_exp = tf.reduce_mean(log_likelihood_samples, axis=0)  # [batch]
        kl_div = tf.reduce_mean(kl_term_samples, axis=0)  # [batch]
        elbo = var_exp - kl_div  # batch
        loss = - tf.reduce_mean(elbo)  # scalar

        entropy = -tf.reduce_sum(logits * token_samples_onehot, axis=-1)  # [S, batch*H*W]
        perplexity = 2. ** (-entropy / tf.math.log(2.))  # [S, batch*H*W]
        mean_perplexity = tf.reduce_mean(perplexity)  # scalar

        if self.step % 10 == 0:
            for i in range(self.num_channels):
                img_mu_0 = tf.reduce_mean(decoded_ims, axis=0)[..., i][..., None]
                img_mu_0 -= tf.reduce_min(img_mu_0)
                img_mu_0 /= tf.reduce_max(img_mu_0)
                tf.summary.image(f'mu_{i}', img_mu_0, step=self.step)

                img_i = img[..., i][..., None]
                img_i = (img_i - tf.reduce_min(img_i)) / (
                        tf.reduce_max(img_i) - tf.reduce_min(img_i))
                tf.summary.image(f'img_before_autoencoder_{i}', img_i, step=self.step)

            # logits = tf.nn.softmax(logits, axis=-1)  # [batch*H*W, num_embedding]
            logits -= tf.reduce_min(logits)
            logits /= tf.reduce_max(logits)
            logits = tf.reshape(logits, [batch, H*W, self.num_embedding])  # [batch, H*W, num_embedding]
            tf.summary.image('logits', logits[:, :, :, None], step=self.step)

            token_sample_onehot = token_samples_onehot[0, ...]
            token_sample_onehot -= tf.reduce_min(token_sample_onehot)  # [batch*H*W, num_embeddings]
            token_sample_onehot /= tf.reduce_max(token_sample_onehot)
            token_sample_onehot = tf.reshape(token_sample_onehot, [batch, H*W, self.num_embedding])  # [batch, H*W, num_embedding]
            tf.summary.image('token_sample_onehot', token_sample_onehot[:, :, :, None], step=self.step)

            latent_img = latent_imgs[:, : , :, 0]
            latent_img -= tf.reduce_min(latent_img)
            latent_img /= tf.reduce_max(latent_img)
            tf.summary.image('latent_im', latent_img[..., None], step=self.step)

        if self.step % 10 == 0:
            tf.summary.scalar('perplexity', mean_perplexity, step=self.step)
            tf.summary.scalar('var_exp', tf.reduce_mean(var_exp), step=self.step)
            tf.summary.scalar('kl_div', tf.reduce_mean(kl_div), step=self.step)
            tf.summary.scalar('temperature', self.temperature, step=self.step)


        return dict(loss=loss,
                    metrics=dict(var_exp=var_exp,
                                 kl_div=kl_div,
                                 mean_perplexity=mean_perplexity))

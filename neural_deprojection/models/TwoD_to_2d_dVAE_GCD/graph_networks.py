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
from tensorflow_addons.image import gaussian_filter2d


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
                 embedding_dim: int = 64,
                 num_embedding: int = 1024,
                 kernel_size: int = 4,
                 num_token_samples: int = 1,
                 num_properties: int = 10,
                 num_channels=1,
                 name=None):
        super(DiscreteImageVAE, self).__init__(name=name)
        # (num_embedding, embedding_dim)
        self.num_channels=num_channels
        self.embeddings = tf.Variable(initial_value=tf.random.truncated_normal((num_embedding, embedding_dim)),
                                      name='embeddings')
        self.num_token_samples = num_token_samples
        self.num_properties = num_properties
        self.num_embedding = num_embedding
        self.embedding_dim = embedding_dim
        self.residual_enc = ResidualStack(num_hiddens=32, num_residual_layers=2, num_residual_hiddens=32)
        self.residual_dec = ResidualStack(num_hiddens=32, num_residual_layers=2, num_residual_hiddens=32)
        self.temperature = tf.Variable(initial_value=tf.constant(1.), name='temperature', trainable=False)
        self.beta = tf.Variable(initial_value=tf.constant(6.6), name='beta', trainable=False)

        self.encoder = snt.Sequential(
            [snt.Conv2D(4, kernel_size, stride=2, padding='SAME', name='conv4'), tf.nn.leaky_relu,  # [b, 512, 512, 4]
             snt.Conv2D(8, kernel_size, stride=2, padding='SAME', name='conv8'), tf.nn.leaky_relu,  # [b, 256, 256, 8]
             snt.Conv2D(16, kernel_size, stride=2, padding='SAME', name='conv16'), tf.nn.leaky_relu,
             # [b, 128, 128, 16]
             snt.Conv2D(32, kernel_size, stride=2, padding='SAME', name='conv32'), tf.nn.leaky_relu,  # [b, 64, 64, 32]
             self.residual_enc,
             snt.Conv2D(64, kernel_size, stride=2, padding='SAME', name='conv64'), tf.nn.leaky_relu,  # [b, 32, 32, 64]
             snt.Conv2D(64, kernel_shape=3, stride=1, padding='SAME', name='conv_enc_1'), tf.nn.leaky_relu,
             snt.Conv2D(self.num_embedding, kernel_shape=6, stride=1, padding='SAME', name='conv_enc_1'), tf.nn.leaky_relu])  # [b, 32, 32, num_embedding]

        # self.VQVAE = snt.nets.VectorQuantizerEMA(embedding_dim=embedding_dim,
        #                                          num_embeddings=num_embedding,
        #                                          commitment_cost=0.25,
        #                                          decay=0.99,
        #                                          name='VQ')

        self.decoder = snt.Sequential(
            [snt.Conv2D(64, kernel_shape=3, stride=1, padding='SAME', name='conv_dec_1'), tf.nn.leaky_relu,
             # [b, 32, 32, 64]
             snt.Conv2DTranspose(32, kernel_size, stride=2, padding='SAME', name='convt32'), tf.nn.leaky_relu,
             # [b, 64, 64, 32]
             self.residual_dec,
             snt.Conv2DTranspose(16, kernel_size, stride=2, padding='SAME', name='convt16'), tf.nn.leaky_relu,
             # [b, 128, 128, 16]
             snt.Conv2DTranspose(8, kernel_size, stride=2, padding='SAME', name='convt8'), tf.nn.leaky_relu,
             # [b, 256, 256, 8]
             snt.Conv2DTranspose(4, kernel_size, stride=2, padding='SAME', name='convt4'), tf.nn.leaky_relu,
             # [b, 512, 512, 4]
             snt.Conv2DTranspose(1, kernel_size, stride=2, padding='SAME', name='convt1'),
             tf.nn.leaky_relu])  # [b, 1024, 1024, 1]

    # @tf.function(input_signature=tf.TensorSpec(shape=[None], dtype=tf.float32))  # what is the shape ???
    # def sample_encoder(self, graph):
    #     return self.encoder(graph)

    def set_beta(self, beta):
        self.beta.assign(beta)

    def set_temperature(self, temperature):
        self.temperature.assign(temperature)

    @tf.function(input_signature=[tf.TensorSpec([None, 3], dtype=tf.float32),
                                  tf.TensorSpec([None, None], dtype=tf.float32),
                                  tf.TensorSpec([], dtype=tf.float32)])
    def sample_decoder(self, logits, temperature):
        token_distribution = tfp.distributions.RelaxedOneHotCategorical(temperature, logits=logits)
        token_samples_onehot = token_distribution.sample((1,),
                                                         name='token_samples')
        token_sample_onehot = token_samples_onehot[0]  # [n_node, num_embedding]
        token_sample = tf.matmul(token_sample_onehot, self.embeddings)  # / self.num_embedding  # [n_node, embedding_dim]  # = z ~ q(z|x)
        latent_img = tf.reshape(token_sample, [1, 32, 32, self.embedding_dim])
        decoded_img = self.decoder(latent_img)  # nodes=[num_gaussian_components, component_dim]
        return decoded_img


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
        logits -= tf.math.reduce_logsumexp(logits, axis=2)  # [batch*H*W, num_embeddings]

        temperature = tf.maximum(0.1, 1. - 0.1/(self.step/1000))
        token_distribution = tfp.distributions.RelaxedOneHotCategorical(temperature, logits=logits)
        token_samples_onehot = token_distribution.sample((self.num_token_samples,), name='token_samples')  # [S, batch*H*W, num_embeddings]

        def _single_decode(token_sample_onehot):
            #[batch*H*W, num_embeddings] @ [num_embeddings, embedding_dim]
            token_sample = tf.matmul(token_sample_onehot, self.embeddings)  # [batch*H*W, embedding_dim]  # = z ~ q(z|x)
            latent_img = tf.reshape(token_sample, [batch, H, W, self.embedding_dim])  # [batch, H, W, embedding_dim]
            decoded_img = self.decoder(latent_img)  # [batch, H', W', C*2]
            img_mu = decoded_img[..., :self.num_channels] #[batch, H', W', C]
            img_logb = decoded_img[..., self.num_channels:]
            log_likelihood = self.log_likelihood(img, img_mu, img_logb)#[batch, H', W', C]
            log_likelihood = tf.reduce_sum(log_likelihood, axis=[-3,-2,-1])  # [batch]
            sum_selected_logits = tf.math.reduce_sum(token_samples_onehot * logits, axis=-1)  # [batch*H*W]
            sum_selected_logits = tf.reshape(sum_selected_logits, [batch, H, W])
            kl_term = tf.reduce_sum(sum_selected_logits, axis=[-2,-1])#[batch]
            return log_likelihood, kl_term

        #num_samples, batch
        log_likelihood_samples, kl_term_samples = tf.vectorized_map(_single_decode, token_samples_onehot)  # [S, batch], [S, batch]

        var_exp = tf.reduce_mean(log_likelihood_samples, axis=0)  # [batch]
        kl_div = tf.reduce_mean(kl_term_samples, axis=0)  # [batch]
        elbo = var_exp - kl_div #batch
        loss = - tf.reduce_mean(elbo)#scalar

        entropy = -tf.reduce_sum(logits * tf.math.exp(logits), axis=-1)  # [batch*H*W]
        perplexity = 2. ** (-entropy / tf.math.log(2.))  # [batch*H*W]
        mean_perplexity = tf.reduce_mean(perplexity)  # scalar


        return dict(loss=loss,
                    metrics=dict(var_exp=var_exp,
                                 kl_div=kl_div,
                                 mean_perplexity=mean_perplexity))

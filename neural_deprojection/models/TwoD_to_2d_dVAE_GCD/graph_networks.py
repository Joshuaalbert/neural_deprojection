import sys

sys.path.insert(1, '/data/s2675544/git/neural_deprojection/')
sys.path.insert(1, '/home/matthijs/git/neural_deprojection/')

from graph_nets import blocks
from graph_nets.utils_tf import concat

import tensorflow as tf
import sonnet as snt
from graph_nets.graphs import GraphsTuple
from graph_nets.utils_tf import fully_connect_graph_dynamic, fully_connect_graph_static
from neural_deprojection.graph_net_utils import AbstractModule, histogramdd
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
                 name=None):
        super(DiscreteImageVAE, self).__init__(name=name)
        # (num_embedding, embedding_dim)
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

    def _build(self, img, **kwargs) -> dict:
        img = gaussian_filter2d(img, filter_shape=[6, 6])[None, :]
        encoded_img = self.encoder(img)
        print(encoded_img.shape)
        # print('\n encoded_graph.nodes', encoded_graph.nodes, '\n')
        # nodes = [n_node, num_embeddings]
        # node = [num_embeddings] -> log(p_i) = logits
        # -> [S, n_node, embedding_dim]
        logits = tf.reshape(encoded_img, [1024, self.num_embedding])  # [n_node, num_embeddings]
        print('\n logits', logits, '\n')
        log_norm = tf.math.reduce_logsumexp(logits, axis=1)  # [n_node]
        print('norm', log_norm)
        if self.step < 1560*20:
            self.set_temperature(1 - 0.95 * tf.cast(self.step, tf.float32) * 3e-5)
        token_distribution = tfp.distributions.RelaxedOneHotCategorical(self.temperature, logits=logits)
        token_samples_onehot = token_distribution.sample((self.num_token_samples,),
                                                         name='token_samples')  # [S, n_node, num_embeddings]
        print('temperature', self.temperature)
        print('token_samples_onehot', token_samples_onehot)

        token_sample = tf.matmul(token_samples_onehot[0], self.embeddings) # / self.num_embedding  # [n_node, embedding_dim]  # = z ~ q(z|x)
        print('token_sample', token_sample)
        latent_img = tf.reshape(token_sample, [32, 32, self.embedding_dim])
        print('latent_img', latent_img)
        decoded_img = self.decoder(latent_img[None, :])  # nodes=[num_gaussian_components, component_dim]
        print('decoded_img', decoded_img)
        log_likelihood = tf.reduce_mean((gaussian_filter2d(img, filter_shape=[6, 6]) - decoded_img) ** 2) # [n_node, num_embeddings].[n_node, num_embeddings]
        # log_likelihood = tf.constant([0.])
        sum_selected_logits = tf.math.reduce_sum(token_samples_onehot[0] * logits, axis=1)  # [n_node]
        print('sum_selected_logits', sum_selected_logits)
        print('log_norm_term', self.num_embedding * log_norm)
        print('embed', self.num_embedding * tf.math.log(tf.cast(self.num_embedding, tf.float32)))
        kl_term = sum_selected_logits - self.num_embedding * log_norm + \
                  self.num_embedding * tf.math.log(tf.cast(self.num_embedding, tf.float32))
        print('kl_term 0', kl_term)
        print('beta', self.beta)
        kl_term = self.beta * tf.reduce_mean(kl_term, axis=0)
        print('log_likelihood', log_likelihood)
        print('kl_term', kl_term)

        # print('vectorized_map',[_single_decode(sample) for sample in token_samples_onehot])
        # log_likelihood_samples, kl_term_samples, decoded_img = tf.vectorized_map(_single_decode, token_samples_onehot)  # [S],[S]
        print('log_likelihood_samples',log_likelihood)
        print('kl_term_samples', kl_term)
        print('Vectorized map works!!')
        # good metric = average entropy of embedding usage! The more precisely embeddings are selected the lower the entropy.

        log_prob_tokens = logits - log_norm[:, None]  # num_tokens, num_embeddings
        entropy = -tf.reduce_sum(log_prob_tokens * tf.math.exp(log_prob_tokens), axis=1)  # num_tokens
        perplexity = 2. ** (-entropy / tf.math.log(2.))
        mean_perplexity = tf.reduce_mean(perplexity)
        print('decoded_img 2', decoded_img)

        var_exp = log_likelihood
        kl_term = kl_term
        # decoded_img = tf.reduce_mean(decoded_img, axis=0)
        # print('decoded_img 3', decoded_img)

        # elbo_samples = log_likelihood_samples - kl_term_samples
        # elbo = tf.reduce_mean(elbo_samples)
        # loss = - elbo  # maximize ELBO so minimize -ELBO
        loss = var_exp

        if self.step % 100 == 0:
            img_before_autoencoder = (img - tf.reduce_min(img)) / (
                    tf.reduce_max(img) - tf.reduce_min(img))

            img_after_autoencoder = (decoded_img - tf.reduce_min(decoded_img)) / (
                    tf.reduce_max(decoded_img) - tf.reduce_min(decoded_img))

            tf.summary.image(f'img_before_autoencoder', img_before_autoencoder, step=self.step)
            tf.summary.image('img_after_autoencoder', img_after_autoencoder, step=self.step)

        if self.step % 10 == 0:
            logits = tf.nn.softmax(logits, axis=-1)
            logits -= tf.reduce_min(logits)
            logits /= tf.reduce_max(logits)
            # tf.repeat(tf.repeat(logits, 16*[4], axis=0), 512*[4], axis=1)
            tf.summary.image('logits', logits[None,:,:,None], step=self.step)
            tf.summary.scalar('perplexity', mean_perplexity, step=self.step)
            tf.summary.scalar('var_exp', var_exp, step=self.step)
            tf.summary.scalar('kl_term', kl_term, step=self.step)

        return dict(loss=loss,
                    metrics=dict(var_exp=var_exp,
                                 kl_term=kl_term,
                                 mean_perplexity=mean_perplexity))


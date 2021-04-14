import sys, glob, os
sys.path.insert(1, '/data/s2675544/git/neural_deprojection/')
import tensorflow as tf
import sonnet as snt
import numpy as np

from neural_deprojection.graph_net_utils import vanilla_training_loop, batch_dataset_set_graph_tuples, \
    TrainOneEpoch, AbstractModule, get_distribution_strategy, build_log_dir, build_checkpoint_dir
from neural_deprojection.models.identify_medium_GCD.main import build_dataset
from tensorflow_addons.image import gaussian_filter2d

class ResidualStack(AbstractModule):
    def __init__(self,
                 num_hiddens,
                 num_residual_layers,
                 num_residual_hiddens,
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
                name="res3x3_%d" % i)
            conv1 = snt.Conv2D(
                output_channels=num_hiddens,
                kernel_shape=(1, 1),
                stride=(1, 1),
                name="res1x1_%d" % i)
            self._layers.append((conv3, conv1))

    def _build(self, inputs):
        h = inputs
        for conv3, conv1 in self._layers:
            conv3_out = conv3(tf.nn.relu(h))
            conv1_out = conv1(tf.nn.relu(conv3_out))
            h += conv1_out
        return tf.nn.relu(h)  # Resnet V1 style

class VariationalAutoEncoder(AbstractModule):
    def __init__(self,
                 n_latent=4,
                 kernel_size=4,
                 name=None):
        super(VariationalAutoEncoder, self).__init__(name=name)

        self.n_latent = n_latent
        self.encoder = snt.Sequential([snt.Conv2D(4, kernel_size, stride=4, padding='SAME'), tf.nn.relu,  # [b, 250, 250, 4]
                                       snt.Conv2D(16, kernel_size, stride=4, padding='SAME'), tf.nn.relu,  # [b, 63, 63, 16]
                                       snt.Conv2D(32, kernel_size, stride=4, padding='SAME'), tf.nn.relu,  # [b, 16, 16, 32]
                                       snt.Conv2D(64, kernel_size, stride=2, padding='SAME'), tf.nn.relu,  # [b, 8, 8, 64]
                                       snt.Flatten()])

        self.mn = snt.nets.MLP([n_latent], activation=tf.nn.relu)
        self.std = snt.nets.MLP([n_latent], activation=tf.nn.relu)

        self.decoder = snt.Sequential([snt.nets.MLP([8*8*64], activation=tf.nn.leaky_relu),
                                       snt.Reshape([8, 8, 64]),
                                       snt.Conv2DTranspose(64, kernel_size, stride=2, padding='SAME'), tf.nn.relu,  # [b, 16, 16, 64]
                                       snt.Conv2DTranspose(32, kernel_size, stride=4, padding='SAME'), tf.nn.relu,  # [b, 64, 64, 32]
                                       snt.Conv2DTranspose(16, kernel_size, stride=4, padding='SAME'), tf.nn.relu,  # [b, 256, 256, 16]
                                       snt.Conv2DTranspose(4, kernel_size, stride=4, padding='SAME'), tf.nn.relu,  # [b, 1024, 1024, 4]
                                       snt.Conv2D(1, kernel_size, padding='SAME')])  # [b, 1024, 1024, 1]

    @property
    def step(self):
        if self._step is None:
            raise ValueError("Need to set step idx variable. model.step = epoch")
        return self._step

    @step.setter
    def step(self, value):
        self._step = value

    def _build(self, batch):
        (img, ) = batch
        img = gaussian_filter2d(img, filter_shape=[6, 6])
        img_before_autoencoder = (img - tf.reduce_min(img)) / (
                tf.reduce_max(img) - tf.reduce_min(img))
        tf.summary.image(f'img_before_autoencoder', img_before_autoencoder, step=self.step)
        encoded_img = self.encoder(img)
        print(encoded_img.shape)

        mn = self.mn(encoded_img)
        std = self.std(encoded_img)
        epsilon = tf.random.normal(tf.stack([tf.shape(encoded_img)[0], self.n_latent]))
        z = mn + tf.multiply(epsilon, tf.exp(tf.multiply(0.5, std)))

        decoded_img = self.decoder(z)
        print(decoded_img.shape)
        img_after_autoencoder = (decoded_img - tf.reduce_min(decoded_img)) / (
                tf.reduce_max(decoded_img) - tf.reduce_min(decoded_img))
        tf.summary.image(f'img_after_autoencoder', img_after_autoencoder, step=self.step)
        return mn, std, z, decoded_img


class VectorQuantizerVariationalAutoEncoder(AbstractModule):
    def __init__(self,
                 embedding_dim=64,
                 num_embeddings=64,
                 kernel_size=4,
                 name=None):
        super(VectorQuantizerVariationalAutoEncoder, self).__init__(name=name)

        self.residual = ResidualStack(num_hiddens=64, num_residual_layers=2, num_residual_hiddens=32)

        self.embedding_dim = embedding_dim
        self.num_embeddings = num_embeddings
        self.encoder = snt.Sequential([snt.Conv2D(4, kernel_size, stride=2, padding='SAME', name='conv4'), tf.nn.relu,  # [b, 500, 500, 4]
                                       snt.Conv2D(8, kernel_size, stride=2, padding='SAME', name='conv8'), tf.nn.relu,  # [b, 250, 250, 8]
                                       snt.Conv2D(16, kernel_size, stride=2, padding='SAME', name='conv16'), tf.nn.relu,  # [b, 125, 125, 16]
                                       snt.Conv2D(32, kernel_size, stride=2, padding='SAME', name='conv32'), tf.nn.relu,  # [b, 63, 63, 32]
                                       snt.Conv2D(64, kernel_size, stride=2, padding='SAME', name='conv64'), tf.nn.relu,  # [b, 32, 32, 64]
                                       self.residual])


        self.VQVAE = snt.nets.VectorQuantizerEMA(embedding_dim=embedding_dim,
                                                 num_embeddings=num_embeddings,
                                                 commitment_cost=0.25,
                                                 decay=0.994413,
                                                 name='VQ')

        self.decoder = snt.Sequential([snt.Conv2DTranspose(32, kernel_size, stride=2, padding='SAME', name='convt32'), tf.nn.relu,  # [b, 64, 64, 32]
                                       snt.Conv2DTranspose(16, kernel_size, stride=2, padding='SAME', name='convt16'), tf.nn.relu,  # [b, 128, 128, 16]
                                       snt.Conv2DTranspose(8, kernel_size, stride=2, padding='SAME', name='convt8'), tf.nn.relu,  # [b, 256, 256, 8]
                                       snt.Conv2DTranspose(4, kernel_size, stride=2, padding='SAME', name='convt4'), tf.nn.relu,  # [b, 512, 512, 4]
                                       snt.Conv2DTranspose(1, kernel_size, stride=2, padding='SAME', name='convt1'), tf.nn.relu,  # [b, 1024, 1024, 1]
                                       snt.Conv2D(1, kernel_size, padding='SAME', name='conv1')])  # [b, 1024, 1024, 1]

    @property
    def step(self):
        if self._step is None:
            raise ValueError("Need to set step idx variable. model.step = epoch")
        return self._step

    @step.setter
    def step(self, value):
        self._step = value

    def _build(self, batch):
        (img, ) = batch
        img = gaussian_filter2d(img, filter_shape=[6, 6])

        img_before_autoencoder = (img - tf.reduce_min(img)) / (
                tf.reduce_max(img) - tf.reduce_min(img))
        tf.summary.image(f'img_before_autoencoder', img_before_autoencoder, step=self.step)

        encoded_img = self.encoder(img)
        vq_dict = self.VQVAE(encoded_img, is_training=True)

        quantized_img = (vq_dict['quantize'] - tf.reduce_min(vq_dict['quantize'])) / (
                tf.reduce_max(vq_dict['quantize']) - tf.reduce_min(vq_dict['quantize']))

        print('SHAPE : ', quantized_img[:, :, :, np.random.randint(low=0, high=32)][:,:,:,None])

        tf.summary.image(f'quantized_img', quantized_img[:, :, :, np.random.randint(low=0, high=32)][:,:,:,None], step=self.step)

        decoded_img = self.decoder(vq_dict['quantize'])

        img_after_autoencoder = (decoded_img - tf.reduce_min(decoded_img)) / (
                tf.reduce_max(decoded_img) - tf.reduce_min(decoded_img))
        tf.summary.image(f'img_after_autoencoder', img_after_autoencoder, step=self.step)

        return vq_dict['loss'], decoded_img

def train_variational_autoencoder(data_dir):
    # strategy = get_distribution_strategy(use_cpus=False, logical_per_physical_factor=1, memory_limit=10000)

    # lists containing tfrecord files
    train_tfrecords = glob.glob(os.path.join(data_dir, 'train', '*.tfrecords'))
    test_tfrecords = glob.glob(os.path.join(data_dir, 'test', '*.tfrecords'))

    train_dataset = build_dataset(train_tfrecords)
    test_dataset = build_dataset(test_tfrecords)

    train_dataset = train_dataset.map(lambda graph, img, c: (img,)).batch(batch_size=32)
    test_dataset = test_dataset.map(lambda graph, img, c: (img,)).batch(batch_size=32)

    # with strategy.scope():
    model = VariationalAutoEncoder(n_latent=4, kernel_size=4)

    learning_rate = 1e-3
    opt = snt.optimizers.Adam(learning_rate)

    def loss(model_outputs, batch):
        (img,) = batch
        mn, std, z, decoded_img = model_outputs
        # reconstruction_loss = tf.reduce_mean(tf.reduce_sum(
        #     keras.losses.binary_crossentropy(img, decoded_img), axis=(1, 2)
        #         ))
        reconstruction_loss = tf.reduce_mean((gaussian_filter2d(img, filter_shape=[6, 6]) - decoded_img[:, 12:-12, 12:-12, :]) ** 2)
        kl_loss = -0.5 * (1 + std - tf.square(mn) - tf.exp(std))
        kl_loss = tf.reduce_mean(tf.reduce_sum(kl_loss, axis=1))
        total_loss = reconstruction_loss + kl_loss
        print(f"recon_loss = {reconstruction_loss}")
        print(f"kl_loss = {kl_loss}")
        return total_loss

    train_one_epoch = TrainOneEpoch(model, loss, opt, strategy=None)

    log_dir = 'VAE_log_dir'
    checkpoint_dir = 'VAE_checkpointing'

    vanilla_training_loop(train_one_epoch=train_one_epoch,
                          training_dataset=train_dataset,
                          test_dataset=test_dataset,
                          num_epochs=50,
                          early_stop_patience=5,
                          checkpoint_dir=checkpoint_dir,
                          log_dir=log_dir,
                          debug=False)


def train_VQVAE(data_dir):
    # strategy = get_distribution_strategy(use_cpus=False, logical_per_physical_factor=1, memory_limit=10000)

    # lists containing tfrecord files
    train_tfrecords = glob.glob(os.path.join(data_dir, 'train', '*.tfrecords'))
    test_tfrecords = glob.glob(os.path.join(data_dir, 'test', '*.tfrecords'))

    train_dataset = build_dataset(train_tfrecords)
    test_dataset = build_dataset(test_tfrecords)

    train_dataset = train_dataset.map(lambda graph, img, c: (img,)).batch(batch_size=32)
    test_dataset = test_dataset.map(lambda graph, img, c: (img,)).batch(batch_size=32)

    # with strategy.scope():
    model = VectorQuantizerVariationalAutoEncoder(embedding_dim=64,
                                                  num_embeddings=1024,
                                                  kernel_size=4)

    learning_rate = 1e-4
    opt = snt.optimizers.Adam(learning_rate)

    def loss(model_outputs, batch):
        (img,) = batch
        vq_loss, decoded_img = model_outputs
        print('im shape', img.shape)
        print('dec im shape', decoded_img.shape)
        # reconstruction_loss = tf.reduce_mean(tf.reduce_sum(
        #     keras.losses.binary_crossentropy(img, decoded_img), axis=(1, 2)
        #         ))
        reconstruction_loss = tf.reduce_mean((gaussian_filter2d(img, filter_shape=[6, 6]) - decoded_img[:, 12:-12, 12:-12, :]) ** 2)
        total_loss = reconstruction_loss + vq_loss
        return total_loss

    train_one_epoch = TrainOneEpoch(model, loss, opt, strategy=None)

    log_dir = 'vqvae_log_dir'
    checkpoint_dir = 'vqvae_checkpointing'

    vanilla_training_loop(train_one_epoch=train_one_epoch,
                          training_dataset=train_dataset,
                          test_dataset=test_dataset,
                          num_epochs=50,
                          early_stop_patience=5,
                          checkpoint_dir=checkpoint_dir,
                          log_dir=log_dir,
                          debug=False)

def main(data_dir):
    # train_variational_autoencoder(data_dir)
    train_VQVAE(data_dir)


if __name__ == '__main__':
    tfrec_base_dir = '/home/s2675544/data/tf_records'
    tfrec_dir = os.path.join(tfrec_base_dir, 'snap_128_tf_records')

    main(tfrec_dir)

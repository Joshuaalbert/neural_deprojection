import sys, glob, os
sys.path.insert(1, '/data/s1825216/git/neural_deprojection/')
import tensorflow as tf
import sonnet as snt
from tensorflow import keras

from neural_deprojection.graph_net_utils import vanilla_training_loop, TrainOneEpoch, AbstractModule, get_distribution_strategy, build_log_dir, build_checkpoint_dir
from neural_deprojection.models.identify_medium_SCD.main import build_dataset


class VariationalAutoEncoder(AbstractModule):
    def __init__(self,
                 n_latent = 4,
                 kernel_size=4,
                 name=None):
        super(VariationalAutoEncoder, self).__init__(name=name)

        self.n_latent = n_latent
        self.encoder = snt.Sequential([snt.Conv2D(4, kernel_size, stride=4, padding='SAME'), tf.nn.leaky_relu,    # [b, 128, 128, 4]
                                       snt.Conv2D(8, kernel_size, stride=2, padding='SAME'), tf.nn.leaky_relu,    # [b, 64, 64, 8]
                                       snt.Conv2D(16, kernel_size, stride=2, padding='SAME'), tf.nn.leaky_relu,   # [b, 32, 32, 16]
                                       snt.Conv2D(32, kernel_size, stride=2, padding='SAME'), tf.nn.leaky_relu,   # [b, 16, 16, 32]
                                       snt.Conv2D(64, kernel_size, stride=2, padding='SAME'), tf.nn.leaky_relu,   # [b, 8, 8, 64]
                                       snt.Flatten()])

        self.mn = snt.nets.MLP([n_latent], activation=tf.nn.leaky_relu)
        self.std = snt.nets.MLP([n_latent], activation=tf.nn.leaky_relu)

        self.decoder = snt.Sequential([snt.nets.MLP([8*8*64], activation=tf.nn.leaky_relu),
                                       snt.Reshape([8, 8, 64]),
                                       snt.Conv2DTranspose(64, kernel_size, stride=2, padding='SAME'), tf.nn.leaky_relu,
                                       snt.Conv2DTranspose(32, kernel_size, stride=2, padding='SAME'), tf.nn.leaky_relu,
                                       snt.Conv2DTranspose(16, kernel_size, stride=2, padding='SAME'), tf.nn.leaky_relu,
                                       snt.Conv2DTranspose(8, kernel_size, stride=2, padding='SAME'), tf.nn.leaky_relu,
                                       snt.Conv2DTranspose(4, kernel_size, stride=2, padding='SAME'), tf.nn.leaky_relu,
                                       snt.Conv2D(1, kernel_size, padding='SAME')])

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
        img_after_autoencoder = (decoded_img - tf.reduce_min(decoded_img)) / (
                tf.reduce_max(decoded_img) - tf.reduce_min(decoded_img))
        tf.summary.image(f'img_after_autoencoder', img_after_autoencoder, step=self.step)
        return mn, std, z, decoded_img


def train_variational_autoencoder(data_dir):
    # strategy = get_distribution_strategy(use_cpus=False, logical_per_physical_factor=1, memory_limit=10000)

    # lists containing tfrecord files
    train_dataset = build_dataset(os.path.join(data_dir, 'train'))
    test_dataset = build_dataset(os.path.join(data_dir, 'test'))

    train_dataset = train_dataset.map(lambda graph, img, c: (img,)).batch(batch_size=32)
    test_dataset = test_dataset.map(lambda graph, img, c: (img,)).batch(batch_size=32)

    # with strategy.scope():
    model = VariationalAutoEncoder(n_latent=2, kernel_size=4)

    learning_rate = 1e-3
    opt = snt.optimizers.Adam(learning_rate)

    def loss(model_outputs, batch):
        (img,) = batch
        mn, std, z, decoded_img = model_outputs
        # reconstruction_loss = tf.reduce_mean(tf.reduce_sum(
        #     keras.losses.binary_crossentropy(img, decoded_img), axis=(1, 2)
        #         ))
        reconstruction_loss = tf.reduce_mean((img - decoded_img[:, :, :, :]) ** 2)
        kl_loss = -0.5 * (1 + std - tf.square(mn) - tf.exp(std))
        kl_loss = tf.reduce_mean(tf.reduce_sum(kl_loss, axis=1))
        total_loss = reconstruction_loss + kl_loss
        return reconstruction_loss

    train_one_epoch = TrainOneEpoch(model, loss, opt, strategy=None)

    log_dir = 'VAE_log_dir'
    checkpoint_dir = 'VAE_checkpointing'

    vanilla_training_loop(train_one_epoch=train_one_epoch,
                          training_dataset=train_dataset,
                          test_dataset=test_dataset,
                          num_epochs=100,
                          early_stop_patience=100,
                          checkpoint_dir=checkpoint_dir,
                          log_dir=log_dir,
                          debug=False)


class VectorQuantizerVariationalAutoEncoder(AbstractModule):
    def __init__(self,
                 n_latent = 4,
                 kernel_size=4,
                 name=None):
        super(VectorQuantizerVariationalAutoEncoder, self).__init__(name=name)

        self.n_latent = n_latent
        self.encoder = snt.Sequential([snt.Conv2D(4, kernel_size, stride=4, padding='SAME'), tf.nn.leaky_relu,    # [b, 128, 128, 4]
                                       snt.Conv2D(8, kernel_size, stride=2, padding='SAME'), tf.nn.leaky_relu,    # [b, 64, 64, 8]
                                       snt.Conv2D(16, kernel_size, stride=2, padding='SAME'), tf.nn.leaky_relu,   # [b, 32, 32, 16]
                                       snt.Conv2D(32, kernel_size, stride=2, padding='SAME'), tf.nn.leaky_relu,   # [b, 16, 16, 32]
                                       snt.Conv2D(64, kernel_size, stride=2, padding='SAME'), tf.nn.leaky_relu])   # [b, 8, 8, 64]

        self.VQVAE = snt.nets.VectorQuantizer(embedding_dim=64, num_embeddings=n_latent, commitment_cost=1)

        self.decoder = snt.Sequential([snt.Conv2DTranspose(64, kernel_size, stride=2, padding='SAME'), tf.nn.leaky_relu,    # [b, 8, 8, 64]
                                       snt.Conv2DTranspose(32, kernel_size, stride=2, padding='SAME'), tf.nn.leaky_relu,    # [b, 32, 32, 32]
                                       snt.Conv2DTranspose(16, kernel_size, stride=2, padding='SAME'), tf.nn.leaky_relu,    # [b, 32, 32, 32]
                                       snt.Conv2DTranspose(8, kernel_size, stride=2, padding='SAME'), tf.nn.leaky_relu,    # [b, 64, 64, 16]
                                       snt.Conv2DTranspose(4, kernel_size, stride=2, padding='SAME'), tf.nn.leaky_relu,     # [b, 128, 128, 8]
                                       snt.Conv2DTranspose(1, kernel_size, stride=2, padding='SAME'), tf.nn.leaky_relu,     # [b, 256, 256, 4]
                                       snt.Conv2D(1, kernel_size, padding='SAME')])    # [b, 256, 256, 1]

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
        img_before_autoencoder = (img - tf.reduce_min(img)) / (
                tf.reduce_max(img) - tf.reduce_min(img))
        tf.summary.image(f'img_before_autoencoder', img_before_autoencoder, step=self.step)
        encoded_img = self.encoder(img)
        print('enc im shape', encoded_img.shape)

        vq_dict = self.VQVAE(encoded_img, is_training=True)

        decoded_img = self.decoder(vq_dict['quantize'])
        img_after_autoencoder = (decoded_img - tf.reduce_min(decoded_img)) / (
                tf.reduce_max(decoded_img) - tf.reduce_min(decoded_img))
        tf.summary.image(f'img_after_autoencoder', img_after_autoencoder, step=self.step)
        return vq_dict['loss'], decoded_img


def train_VQVAE(data_dir):
    # strategy = get_distribution_strategy(use_cpus=False, logical_per_physical_factor=1, memory_limit=10000)

    # lists containing tfrecord files
    train_dataset = build_dataset(os.path.join(data_dir, 'train'))
    test_dataset = build_dataset(os.path.join(data_dir, 'test'))

    train_dataset = train_dataset.map(lambda graph, img, c: (img,)).batch(batch_size=32)
    test_dataset = test_dataset.map(lambda graph, img, c: (img,)).batch(batch_size=32)

    # with strategy.scope():
    model = VectorQuantizerVariationalAutoEncoder(n_latent=4, kernel_size=4)

    learning_rate = 1e-3
    opt = snt.optimizers.Adam(learning_rate)

    def loss(model_outputs, batch):
        (img,) = batch
        vq_loss, decoded_img = model_outputs
        print('im shape', img.shape)
        print('dec im shape', decoded_img.shape)
        # reconstruction_loss = tf.reduce_mean(tf.reduce_sum(
        #     keras.losses.binary_crossentropy(img, decoded_img), axis=(1, 2)
        #         ))
        reconstruction_loss = tf.reduce_mean((img - decoded_img[:, :, :, :]) ** 2)
        total_loss = reconstruction_loss + vq_loss
        return total_loss

    train_one_epoch = TrainOneEpoch(model, loss, opt, strategy=None)

    log_dir = 'VQVAE_log_dir'
    checkpoint_dir = 'VQVAE_checkpointing'

    vanilla_training_loop(train_one_epoch=train_one_epoch,
                          training_dataset=train_dataset,
                          test_dataset=test_dataset,
                          num_epochs=100,
                          early_stop_patience=100,
                          checkpoint_dir=checkpoint_dir,
                          log_dir=log_dir,
                          debug=False)


if __name__ == '__main__':
    test_train_dir = '/home/s1825216/data/train_data/ClaudeData/'
    # test_train_dir = '/home/s1825216/data/train_data/auto_encoder/'
    # train_variational_autoencoder(test_train_dir)
    train_VQVAE(test_train_dir)
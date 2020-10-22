import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import tensorflow_addons as tfa

OUTPUT_CHANNELS = 3


def downsample(filters, size, ds_activation, apply_instancenorm=True):
    initializer = tf.random_normal_initializer(0., 0.02)
    gamma_init = keras.initializers.RandomNormal(mean=0.0, stddev=0.02)

    result = keras.Sequential()
    result.add(layers.Conv2D(filters, size, strides=2, padding='same',
                             kernel_initializer=initializer, use_bias=False))

    if apply_instancenorm:
        result.add(tfa.layers.InstanceNormalization(gamma_initializer=gamma_init))

    result.add(ds_activation)

    return result


def upsample(filters, size, us_activation, apply_dropout=False):
    initializer = tf.random_normal_initializer(0., 0.02)
    gamma_init = keras.initializers.RandomNormal(mean=0.0, stddev=0.02)

    result = keras.Sequential()
    result.add(layers.Conv2DTranspose(filters, size, strides=2,
                                      padding='same',
                                      kernel_initializer=initializer,
                                      use_bias=False))

    result.add(tfa.layers.InstanceNormalization(gamma_initializer=gamma_init))

    if apply_dropout:
        result.add(layers.Dropout(0.5))

    result.add(us_activation)

    return result


def Generator(us_activation, ds_activation, kernel_size):
    inputs = layers.Input(shape=[256, 256, 3])

    # bs = batch size
    down_stack = [
        downsample(64, size=kernel_size, apply_instancenorm=False, ds_activation=ds_activation),  # (bs, 128, 128, 64)
        downsample(128, size=kernel_size, ds_activation=ds_activation),  # (bs, 64, 64, 128)
        downsample(256, size=kernel_size, ds_activation=ds_activation),  # (bs, 32, 32, 256)
        downsample(512, size=kernel_size, ds_activation=ds_activation),  # (bs, 16, 16, 512)
        downsample(512, size=kernel_size, ds_activation=ds_activation),  # (bs, 8, 8, 512)
        downsample(512, size=kernel_size, ds_activation=ds_activation),  # (bs, 4, 4, 512)
        downsample(512, size=kernel_size, ds_activation=ds_activation),  # (bs, 2, 2, 512)
        downsample(512, size=kernel_size, ds_activation=ds_activation),  # (bs, 1, 1, 512)
    ]

    up_stack = [
        upsample(512, size=kernel_size, apply_dropout=True, us_activation=us_activation),  # (bs, 2, 2, 1024)
        upsample(512, size=kernel_size, apply_dropout=True, us_activation=us_activation),  # (bs, 4, 4, 1024)
        upsample(512, size=kernel_size, apply_dropout=True, us_activation=us_activation),  # (bs, 8, 8, 1024)
        upsample(512, size=kernel_size, us_activation=us_activation),  # (bs, 16, 16, 1024)
        upsample(256, size=kernel_size, us_activation=us_activation),  # (bs, 32, 32, 512)
        upsample(128, size=kernel_size, us_activation=us_activation),  # (bs, 64, 64, 256)
        upsample(64, size=kernel_size, us_activation=us_activation),  # (bs, 128, 128, 128)
    ]

    initializer = tf.random_normal_initializer(0., 0.02)
    last = layers.Conv2DTranspose(OUTPUT_CHANNELS, 4,
                                  strides=2,
                                  padding='same',
                                  kernel_initializer=initializer,
                                  activation='tanh')  # (bs, 256, 256, 3)

    x = inputs

    # Downsampling through the model
    skips = []
    for down in down_stack:
        x = down(x)
        skips.append(x)

    skips = reversed(skips[:-1])

    # Upsampling and establishing the skip connections
    for up, skip in zip(up_stack, skips):
        x = up(x)
        x = layers.Concatenate()([x, skip])

    x = last(x)

    return keras.Model(inputs=inputs, outputs=x)


def Discriminator(ds_activation, kernel_size):
    initializer = tf.random_normal_initializer(0., 0.02)
    gamma_init = keras.initializers.RandomNormal(mean=0.0, stddev=0.02)

    inp = layers.Input(shape=[256, 256, 3], name='input_image')

    x = inp

    down1 = downsample(64, size=kernel_size, apply_instancenorm=False, ds_activation=ds_activation)(x)  # (bs, 128, 128, 64)
    down2 = downsample(128, size=kernel_size, ds_activation=ds_activation)(down1)  # (bs, 64, 64, 128)
    down3 = downsample(256, size=kernel_size, ds_activation=ds_activation)(down2)  # (bs, 32, 32, 256)

    zero_pad1 = layers.ZeroPadding2D()(down3)  # (bs, 34, 34, 256)
    conv = layers.Conv2D(512, 4, strides=1,
                         kernel_initializer=initializer,
                         use_bias=False)(zero_pad1)  # (bs, 31, 31, 512)

    norm1 = tfa.layers.InstanceNormalization(gamma_initializer=gamma_init)(conv)

    leaky_relu = layers.LeakyReLU()(norm1)

    zero_pad2 = layers.ZeroPadding2D()(leaky_relu)  # (bs, 33, 33, 512)

    last = layers.Conv2D(1, 4, strides=1,
                         kernel_initializer=initializer)(zero_pad2)  # (bs, 30, 30, 1)

    return tf.keras.Model(inputs=inp, outputs=last)

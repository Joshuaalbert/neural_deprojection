import argparse,  sys
#imports
import tensorflow as tf
import tensorflow_addons as tfa
from tensorflow.keras import layers
import matplotlib.pyplot as plt
import numpy as np
import PIL
import shutil
import os

#import py files
from neural_deprojection.monet.read_tfrec import load_dataset
from neural_deprojection.monet.build_gen_dis import Generator, Discriminator
from neural_deprojection.monet.cycle_gan import CycleGan, build_discriminator_loss, build_generator_loss,\
    build_calc_cycle_loss, build_identity_loss

# hyperparameters


def main(data_dir, lr, optimizer, ds_activation, us_activation, kernel_size, sync_period):
    """

    Args:
        lr:
        optimizer:
        ds_activation:
        us_activation:
        kernel_size:
        sync_period:

    Returns:

    """

    os.makedirs('./images', exist_ok=True)

    if optimizer == 'adam':
        optimizer = tf.keras.optimizers.Adam(learning_rate=lr, beta_1=0.5)
    elif optimizer == 'ranger':
        optimizer = tfa.optimizers.Lookahead(tfa.optimizers.RectifiedAdam(lr, beta_1=0.5), sync_period=sync_period)
    if ds_activation == 'leaky_relu':
        ds_activation = layers.LeakyReLU()
    elif ds_activation == 'relu':
        ds_activation = layers.ReLU()
    elif ds_activation == 'mish':
        ds_activation = layers.Lambda(lambda x : x * tf.math.tanh(tf.nn.softplus(x)))
    else:
        raise ValueError(f"{ds_activation} doesn't exist")
    if us_activation == 'leaky_relu':
        us_activation = layers.LeakyReLU()
    elif us_activation == 'relu':
        us_activation = layers.ReLU()
    elif us_activation == 'mish':
        us_activation = layers.Lambda(lambda x : x * tf.math.tanh(tf.nn.softplus(x)))
    else:
        raise ValueError(f"{us_activation} doesn't exist")

    #Try to use a TPU if available
    try:
        tpu = tf.distribute.cluster_resolver.TPUClusterResolver()
        print('Device:', tpu.master())
        tf.config.experimental_connect_to_cluster(tpu)
        tf.tpu.experimental.initialize_tpu_system(tpu)
        strategy = tf.distribute.experimental.TPUStrategy(tpu)
    except:
        strategy = tf.distribute.get_strategy()
    print('Number of replicas:', strategy.num_replicas_in_sync)

    AUTOTUNE = tf.data.experimental.AUTOTUNE

    print(tf.__version__)

    # LOAD DATA
    MONET_FILENAMES = tf.io.gfile.glob(str(os.path.join(data_dir, 'monet_tfrec/*.tfrec')))
    print('Monet TFRecord Files:', len(MONET_FILENAMES))

    PHOTO_FILENAMES = tf.io.gfile.glob(str(os.path.join(data_dir,'photo_tfrec/*.tfrec')))
    print('Photo TFRecord Files:', len(PHOTO_FILENAMES))

    monet_ds = load_dataset(MONET_FILENAMES, labeled=True, AUTOTUNE=AUTOTUNE).batch(1)
    photo_ds = load_dataset(PHOTO_FILENAMES, labeled=True, AUTOTUNE=AUTOTUNE).batch(1)

    example_monet = next(iter(monet_ds))
    example_photo = next(iter(photo_ds))

    plt.subplot(121)
    plt.title('Photo')
    plt.imshow(example_photo[0] * 0.5 + 0.5)

    plt.subplot(122)
    plt.title('Monet')
    plt.imshow(example_monet[0] * 0.5 + 0.5)
    plt.show()

    # CONSTRUCT MODEL

    with strategy.scope():
        monet_generator = Generator(us_activation=us_activation, ds_activation=ds_activation, kernel_size=kernel_size)  # transforms photos to Monet-esque paintings
        photo_generator = Generator(us_activation=us_activation, ds_activation=ds_activation, kernel_size=kernel_size)  # transforms Monet paintings to be more like photos

        monet_discriminator = Discriminator(ds_activation=ds_activation, kernel_size=kernel_size)  # differentiates real Monet paintings and generated Monet paintings
        photo_discriminator = Discriminator(ds_activation=ds_activation, kernel_size=kernel_size)  # differentiates real photos and generated photos

    to_monet = monet_generator(example_photo)

    plt.subplot(1, 2, 1)
    plt.title("Original Photo")
    plt.imshow(example_photo[0] * 0.5 + 0.5)

    plt.subplot(1, 2, 2)
    plt.title("Monet-esque Photo")
    plt.imshow(to_monet[0] * 0.5 + 0.5)
    plt.show()

    # TRAINING
    with strategy.scope():
        monet_generator_optimizer = optimizer
        photo_generator_optimizer = optimizer

        monet_discriminator_optimizer = optimizer
        photo_discriminator_optimizer = optimizer

    with strategy.scope():
        cycle_gan_model = CycleGan(
            monet_generator, photo_generator, monet_discriminator, photo_discriminator
        )

        cycle_gan_model.compile(
            m_gen_optimizer=monet_generator_optimizer,
            p_gen_optimizer=photo_generator_optimizer,
            m_disc_optimizer=monet_discriminator_optimizer,
            p_disc_optimizer=photo_discriminator_optimizer,
            gen_loss_fn=build_generator_loss(strategy),
            disc_loss_fn=build_discriminator_loss(strategy),
            cycle_loss_fn=build_calc_cycle_loss(strategy),
            identity_loss_fn=build_identity_loss(strategy)
        )

    cycle_gan_model.fit(
        tf.data.Dataset.zip((monet_ds, photo_ds)),
        epochs=25
    )

    _, ax = plt.subplots(5, 2, figsize=(12, 12))
    for i, img in enumerate(photo_ds.take(5)):
        prediction = monet_generator(img, training=False)[0].numpy()
        prediction = (prediction * 127.5 + 127.5).astype(np.uint8)
        img = (img[0] * 127.5 + 127.5).numpy().astype(np.uint8)

        ax[i, 0].imshow(img)
        ax[i, 1].imshow(prediction)
        ax[i, 0].set_title("Input Photo")
        ax[i, 1].set_title("Monet-esque")
        ax[i, 0].axis("off")
        ax[i, 1].axis("off")
    plt.show()

    # SAVE MODEL

    i = 1
    for img in photo_ds:
        prediction = monet_generator(img, training=False)[0].numpy()
        prediction = (prediction * 127.5 + 127.5).astype(np.uint8)
        im = PIL.Image.fromarray(prediction)
        im.save("images/" + str(i) + ".jpg")
        i += 1

        # shutil.make_archive("/kaggle/working/images", 'zip', "/kaggle/images") ???????????????


def add_args(parser):
    parser.register("type", "bool", lambda v: v.lower() == "true")
    #lr, optimizer, ds_activation, us_activation, kernel_size, sync_period
    parser.add_argument('--data_dir', help='Where monet data is stored', type=str, required=True)
    parser.add_argument('--lr', help='Which learning rate to use',default=1e-2, type=float, required=False)
    parser.add_argument('--optimizer', help='Which optimizer to use', default='ranger', type=str, required=False)
    parser.add_argument('--ds_activation', help='Which downsample activation function to use', default='mish',
                        type=str, required=False)
    parser.add_argument('--us_activation', help='Which upsample activation function to use', default='mish',
                        type=str, required=False)
    parser.add_argument('--kernel_size', help='Which kernel size to use', default=4, type=int, required=False)
    parser.add_argument('--sync_period', help='Which sync period to use with ranger optimizer', default=20, type=int,
                        required=False)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Runs simple training loop.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    add_args(parser)
    flags, unparsed = parser.parse_known_args()
    print("Running with:")
    for option, value in vars(flags).items():
        print("    {} -> {}".format(option, value))
    main(**vars(flags))
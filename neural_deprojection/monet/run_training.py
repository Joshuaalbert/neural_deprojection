import argparse,  sys
#imports
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import PIL
import shutil

#import py files
from neural_deprojection.monet.read_tfrec import load_dataset
from neural_deprojection.monet.build_gen_dis import Generator, Discriminator
from neural_deprojection.monet.cycle_gan import CycleGan, discriminator_loss, generator_loss, calc_cycle_loss,\
    identity_loss

def main(arg0, arg1, arg2):
    '''

    Args:
        arg0:
        arg1:
        arg2:

    Returns:

    '''

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
    MONET_FILENAMES = tf.io.gfile.glob(str('data/monet_tfrec/*.tfrec'))
    print('Monet TFRecord Files:', len(MONET_FILENAMES))

    PHOTO_FILENAMES = tf.io.gfile.glob(str('data/photo_tfrec/*.tfrec'))
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
        monet_generator = Generator()  # transforms photos to Monet-esque paintings
        photo_generator = Generator()  # transforms Monet paintings to be more like photos

        monet_discriminator = Discriminator()  # differentiates real Monet paintings and generated Monet paintings
        photo_discriminator = Discriminator()  # differentiates real photos and generated photos

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
        monet_generator_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)
        photo_generator_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)

        monet_discriminator_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)
        photo_discriminator_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)

    with strategy.scope():
        cycle_gan_model = CycleGan(
            monet_generator, photo_generator, monet_discriminator, photo_discriminator
        )

        cycle_gan_model.compile(
            m_gen_optimizer=monet_generator_optimizer,
            p_gen_optimizer=photo_generator_optimizer,
            m_disc_optimizer=monet_discriminator_optimizer,
            p_disc_optimizer=photo_discriminator_optimizer,
            gen_loss_fn=generator_loss,
            disc_loss_fn=discriminator_loss,
            cycle_loss_fn=calc_cycle_loss,
            identity_loss_fn=identity_loss
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
    parser.add_argument('--arg0', help='Obs number L*',default=0, type=int, required=False)
    parser.add_argument('--arg1', help='Obs number L*',default=0, type=int, required=False)
    parser.add_argument('--arg2', help='Obs number L*',default=0, type=int, required=False)

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
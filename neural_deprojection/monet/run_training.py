import argparse
# imports
import tensorflow as tf
import tensorflow_addons as tfa
from tensorflow.keras import layers
import numpy as np
import PIL
import os
import GPyOpt

# import locally (note no neural_deprojection before them)
from read_tfrec import load_dataset
from build_gen_dis import Generator, Discriminator
from cycle_gan import CycleGan, build_discriminator_loss, build_generator_loss, \
    build_calc_cycle_loss, build_identity_loss
from mi_fid import mi_fid


def save_predicted_test_images(monet_generator, test_photo_ds):
    # SAVE MODEL
    i = 1
    for img in test_photo_ds:
        prediction = monet_generator(img, training=False)[0].numpy()
        prediction = (prediction * 127.5 + 127.5).astype(np.uint8)
        im = PIL.Image.fromarray(prediction)
        im.save("images/" + str(i) + ".jpg")
        i += 1

        # shutil.make_archive("/kaggle/working/images", 'zip', "/kaggle/images") ???????????????


def main(num_folds, data_dir, gen_lr, disc_lr, optimizer, ds_activation, us_activation, kernel_size, sync_period, batch_size,
         num_epochs):
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
    print(f"Will perform {num_folds}-fold cross validation.")
    os.makedirs('./images', exist_ok=True)

    AUTOTUNE = tf.data.experimental.AUTOTUNE

    print("TF version:", tf.__version__)

    # LOAD DATA
    MONET_FILENAMES = tf.io.gfile.glob(str(os.path.join(data_dir, 'monet_tfrec/*.tfrec')))
    print('Monet TFRecord Files:', len(MONET_FILENAMES))

    PHOTO_FILENAMES = tf.io.gfile.glob(str(os.path.join(data_dir, 'photo_tfrec/*.tfrec')))
    print('Photo TFRecord Files:', len(PHOTO_FILENAMES))

    def run_fold_training(k_fold, num_folds,
                          optimizer=optimizer,
                          ds_activation=ds_activation,
                          us_activation=us_activation):
        # CONSTRUCT MODEL
        tf.keras.backend.clear_session()
        # Try to use a TPU if available
        try:
            tpu = tf.distribute.cluster_resolver.TPUClusterResolver()
            print('Device:', tpu.master())
            tf.config.experimental_connect_to_cluster(tpu)
            tf.tpu.experimental.initialize_tpu_system(tpu)
            strategy = tf.distribute.experimental.TPUStrategy(tpu)
        except:
            strategy = tf.distribute.get_strategy()
        print('Number of replicas:', strategy.num_replicas_in_sync)

        with strategy.scope():
            if optimizer == 'adam':
                gen_optimizer = tf.keras.optimizers.Adam(learning_rate=gen_lr)
                disc_optimizer = tf.keras.optimizers.Adam(learning_rate=disc_lr)
            elif optimizer == 'ranger':
                gen_optimizer = tfa.optimizers.Lookahead(tfa.optimizers.RectifiedAdam(gen_lr), sync_period=sync_period)
                disc_optimizer = tfa.optimizers.Lookahead(tfa.optimizers.RectifiedAdam(disc_lr), sync_period=sync_period)
            if ds_activation == 'leaky_relu':
                ds_activation = layers.LeakyReLU()
            elif ds_activation == 'relu':
                ds_activation = layers.ReLU()
            elif ds_activation == 'mish':
                ds_activation = layers.Lambda(lambda x: x * tf.math.tanh(tf.nn.softplus(x)))
            else:
                raise ValueError(f"{ds_activation} doesn't exist")
            if us_activation == 'leaky_relu':
                us_activation = layers.LeakyReLU()
            elif us_activation == 'relu':
                us_activation = layers.ReLU()
            elif us_activation == 'mish':
                us_activation = layers.Lambda(lambda x: x * tf.math.tanh(tf.nn.softplus(x)))
            else:
                raise ValueError(f"{us_activation} doesn't exist")

            train_monet_ds = load_dataset(MONET_FILENAMES, labeled=True, AUTOTUNE=AUTOTUNE)
            train_photo_ds = load_dataset(PHOTO_FILENAMES, labeled=True, AUTOTUNE=AUTOTUNE)
            train_images_ds = tf.data.Dataset.zip((train_monet_ds, train_photo_ds)).enumerate().filter(
                lambda i, images, num_folds=num_folds, k_fold=k_fold: i % num_folds != k_fold).map(
                lambda i, images: images).batch(batch_size)

            test_monet_ds = load_dataset(MONET_FILENAMES, labeled=True, AUTOTUNE=AUTOTUNE)
            test_photo_ds = load_dataset(PHOTO_FILENAMES, labeled=True, AUTOTUNE=AUTOTUNE)
            test_images_ds = tf.data.Dataset.zip((test_monet_ds, test_photo_ds)).enumerate().filter(
                lambda i, images, num_folds=num_folds, k_fold=k_fold: i % num_folds == k_fold).map(
                lambda i, images: images).batch(batch_size)

            monet_generator = Generator(us_activation=us_activation, ds_activation=ds_activation,
                                        kernel_size=kernel_size)  # transforms photos to Monet-esque paintings
            photo_generator = Generator(us_activation=us_activation, ds_activation=ds_activation,
                                        kernel_size=kernel_size)  # transforms Monet paintings to be more like photos

            monet_discriminator = Discriminator(ds_activation=ds_activation,
                                                kernel_size=kernel_size)  # differentiates real Monet paintings and generated Monet paintings
            photo_discriminator = Discriminator(ds_activation=ds_activation,
                                                kernel_size=kernel_size)  # differentiates real photos and generated photos

            monet_generator_optimizer = gen_optimizer
            photo_generator_optimizer = gen_optimizer

            monet_discriminator_optimizer = disc_optimizer
            photo_discriminator_optimizer = disc_optimizer

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

            cycle_gan_model.fit(train_images_ds,
                epochs=num_epochs,
                callbacks=[tf.keras.callbacks.EarlyStopping(monitor='total_loss',patience=3, mode='min',
                                                            restore_best_weights=True)])

            # output = cycle_gan_model.evaluate(test_images_ds, return_dict=True)
            mifid_metric = mi_fid(test_images_ds, monet_generator)
        # save_predicted_test_images(monet_generator, test_photo_ds.batch(1))

        return mifid_metric

    cv_loss = sum([run_fold_training(k, num_folds) for k in range(num_folds)]) / num_folds
    print(f"Final {num_folds}-fold cross validation score is: {cv_loss}")
    # TODO(MVG,Hendrix): use cv_loss as a metric for Bayesian optimisation with GPyOpt
    # TODO(MVG,Hendrix): expose a different learning rate for discriminator and generator (so two in total)
    # TODO(MVG,Hendrix): expose cycle_lambda as a hyperparam
    return cv_loss


def func_to_minimize(x):
    num_folds = 3
    data_dir = 'data/'
    num_epochs = 25

    gen_lr = x[0][0]
    disc_lr = x[0][1]

    if x[0][2] == 0:
        optimizer = 'adam'
    elif x[0][2] == 1:
        optimizer = 'ranger'

    if x[0][3] == 0:
        ds_activation = 'relu'
    elif x[0][3] == 1:
        ds_activation = 'leaky_relu'
    elif x[0][3] == 2:
        ds_activation = 'mish'

    if x[0][4] == 0:
        us_activation = 'relu'
    elif x[0][4] == 1:
        us_activation = 'leaky_relu'
    elif x[0][4] == 2:
        us_activation = 'mish'

    kernel_size = int(x[0][5])
    sync_period = int(x[0][6])
    batch_size = int(x[0][7])
    cv_score = main(num_folds, data_dir, gen_lr, disc_lr, optimizer, ds_activation, us_activation, kernel_size, sync_period,
                    batch_size, num_epochs)
    return cv_score

def bayesian_optimizer():
    print('Bayesian Optimization...')
    logresults = open('log_bayesian_results.txt', 'a')
    logresults.write('')
    # make a loop to run multiple times

    bounds2d = [{'name': 'gen_lr', 'type': 'continuous', 'domain': (1e-3, 0.1)},
                {'name': 'disc_lr', 'type': 'continuous', 'domain': (1e-3, 0.1)},
                {'name': 'optimizer', 'type': 'discrete', 'domain': (0, 1)},
                {'name': 'ds_activation', 'type': 'discrete', 'domain': (0, 1, 2)},
                {'name': 'us_activation', 'type': 'discrete', 'domain': (0, 1, 2)},
                {'name': 'kernel_size', 'type': 'discrete', 'domain': (3, 4, 5)},
                {'name': 'sync_period', 'type': 'discrete', 'domain': (10, 20, 30)},
                {'name': 'batch_size', 'type': 'discrete', 'domain': (2, 4, 8)},
                ]

    maxiter = 10

    myBopt_2d = GPyOpt.methods.BayesianOptimization(func_to_minimize, domain=bounds2d)
    myBopt_2d.run_optimization(max_iter=maxiter)
    print("=" * 20)
    print("Value that minimises the objective:" + str(myBopt_2d.x_opt))
    print("Minimum value of the objective: " + str(myBopt_2d.fx_opt))
    print("=" * 20)

    logresults.write("Value that minimises the objective:" + str(myBopt_2d.x_opt))
    logresults.write('\n')
    logresults.write("Minimum value of the objective: " + str(myBopt_2d.fx_opt))
    logresults.write('\n')
    logresults.write('\n')
    logresults.close()


def add_args(parser):
    parser.register("type", "bool", lambda v: v.lower() == "true")
    parser.register("type", "path", lambda v: os.path.expanduser(v))
    # lr, optimizer, ds_activation, us_activation, kernel_size, sync_period
    parser.add_argument('--data_dir', help='Where monet data is stored', type='path', required=True)
    parser.add_argument('--num_folds', help='How many folds of K-folds CV to do.', default=3, type=int, required=False)
    parser.add_argument('--batch_size', help='Batch size of training and evaluation.', default=2, type=int,
                        required=False)
    parser.add_argument('--num_epochs', help='How many epochs to run.', default=25, type=int,
                        required=False)
    parser.add_argument('--gen_lr', help='Which generator learning rate to use', default=1e-2, type=float, required=False)
    parser.add_argument('--disc_lr', help='Which discriminator learning rate to use', default=1e-2, type=float, required=False)
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

    bayesian_optimizer()
    #main(**vars(flags))

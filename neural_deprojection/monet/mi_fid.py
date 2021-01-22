import tensorflow as tf
import tensorflow_probability as tfp

IMAGE_SIZE = [256, 256]

def msqrt(A):
    s, U, Vh = tf.linalg.svd(A)
    s = tf.where(s < 0., 0., s)
    L = U * tf.math.sqrt(s)
    return L

def test_msqrt():
    for i in range(10):
        A = 5*tf.random.normal(shape=(20,20), dtype=tf.float64)
        A = A @ tf.transpose(A)
        L = msqrt(A)
        assert tf.reduce_all(tf.math.abs(A - L @ tf.transpose(L) ) < 1e-6)

def mi_fid(test_ds, gen_model):
    """
    Implements Memorization-informed Fréchet Inception Distance

    Args:
        test_ds: Dataset of (monet, photo)
    Returns:

    """
    inception = tf.keras.applications.InceptionV3(include_top=False, input_shape=[256, 256, 3], pooling='max')
    i = tf.constant(0.)

    all_monet_features = []
    all_monet_gen_features = []
    for (monet, photo) in test_ds:
        monet_features = inception(monet)
        all_monet_features.append(monet_features)
        gen_monet = gen_model(photo)
        gen_monet_features = inception(gen_monet)
        all_monet_gen_features.append(gen_monet_features)
        i += 1

    all_monet_features = tf.concat(all_monet_features, axis=0)
    all_monet_gen_features = tf.concat(all_monet_gen_features, axis=0)

    mu_monet = tf.reduce_mean(all_monet_features, axis=0)
    mu_monet_gen = tf.reduce_mean(all_monet_gen_features, axis=0)
    cov_monet = tfp.stats.covariance(all_monet_features, sample_axis=0)
    cov_monet_gen = tfp.stats.covariance(all_monet_gen_features, sample_axis=0)

    diff = mu_monet - mu_monet_gen

    fid = tf.reduce_sum(tf.square(diff)) + tf.linalg.trace(cov_monet + cov_monet_gen - 2.*msqrt(cov_monet@cov_monet_gen))


    #n_gen, n_real
    cosine_dist = tf.vectorized_map(lambda G: tf.vectorized_map(lambda R: 1. - tf.reduce_sum(G*R)/(tf.linalg.norm(G)*tf.linalg.norm(R)), all_monet_features), all_monet_gen_features)
    mi = tf.reduce_mean(tf.reduce_min(cosine_dist, axis=1),axis=0)
    mi = tf.where(mi > 0.1, 1., mi)

    return fid / mi

def test_mi_fid():

    def mi_fid(monet_ds, gen_monet_ds):
        inception = tf.keras.applications.InceptionV3(include_top=False, input_shape=[256, 256, 3], pooling='avg')
        i = tf.constant(0.)

        all_monet_features = []
        all_monet_gen_features = []
        for monet, gen_monet in zip(monet_ds, gen_monet_ds):
            if i % 2 == 0:
                monet_features = inception(monet)
                all_monet_features.append(monet_features)
            else:
                gen_monet = monet
                gen_monet_features = inception(gen_monet)
                all_monet_gen_features.append(gen_monet_features)
            i += 1
        print("Processed {} monet photos".format(i))

        all_monet_features = tf.concat(all_monet_features, axis=0)
        print("Monet features shape: {}".format(all_monet_features.shape))
        all_monet_gen_features = tf.concat(all_monet_gen_features, axis=0)
        print("Gen monet features shape: {}".format(all_monet_features.shape))

        mu_monet = tf.reduce_mean(all_monet_features, axis=0)
        mu_monet_gen = tf.reduce_mean(all_monet_gen_features, axis=0)
        print("Monet mean", mu_monet)
        print("Gen Monet mean", mu_monet_gen)
        cov_monet = tfp.stats.covariance(all_monet_features, sample_axis=0)
        cov_monet_gen = tfp.stats.covariance(all_monet_gen_features, sample_axis=0)


        print("Monet cov", cov_monet)
        print("Gen Monet cov", cov_monet_gen)
        import pylab as plt
        plt.plot(mu_monet)
        plt.plot(mu_monet_gen)
        plt.show()
        plt.imshow(cov_monet)
        plt.colorbar()
        plt.show()
        plt.imshow(cov_monet_gen)
        plt.colorbar()
        plt.show()

        diff = mu_monet - mu_monet_gen

        fid = tf.reduce_sum(tf.square(diff)) + tf.linalg.trace(
            cov_monet + cov_monet_gen - 2. * msqrt(cov_monet @ cov_monet_gen))

        # n_gen, n_real
        cosine_dist = tf.vectorized_map(
            lambda G: tf.vectorized_map(lambda R: 1. - tf.reduce_sum(G * R) / (tf.linalg.norm(G) * tf.linalg.norm(R)),
                                        all_monet_features), all_monet_gen_features)
        mi = tf.reduce_mean(tf.reduce_min(cosine_dist, axis=1), axis=0)
        mi = tf.where(mi > 0.1, 1., mi)
        print(mi)
        return fid / mi

    import os
    data_dir='~/data/monet/data'
    MONET_FILENAMES = tf.io.gfile.glob(str(os.path.join(os.path.expanduser(data_dir), 'monet_tfrec/*.tfrec')))
    print('Monet TFRecord Files:', len(MONET_FILENAMES))
    PHOTO_FILENAMES = tf.io.gfile.glob(str(os.path.join(os.path.expanduser(data_dir), 'photo_tfrec/*.tfrec')))
    print('Photo TFRecord Files:', len(PHOTO_FILENAMES))
    from read_tfrec import load_dataset
    monet_ds = load_dataset(MONET_FILENAMES, labeled=True, AUTOTUNE=tf.data.experimental.AUTOTUNE).batch(1)
    photo_ds = load_dataset(PHOTO_FILENAMES, labeled=True, AUTOTUNE=tf.data.experimental.AUTOTUNE).batch(1)
    print(mi_fid(monet_ds, photo_ds))





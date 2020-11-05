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




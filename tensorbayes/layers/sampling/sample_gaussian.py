import tensorflow as tf


def sample_gaussian(mean, var, scope):
    """Sample from a Gaussian random variable with a specified mean and
    variance.
    """
    with tf.name_scope(scope):
        # N.B.: Like Scipy, sampling from a random normal requires the standard
        #       deviation in place of the variance. Hence the square root.
        return tf.random_normal(tf.shape(mean), mean, tf.sqrt(var))

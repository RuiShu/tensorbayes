import tensorflow as tf

def gaussian_sample(mean, var, scope):
    with tf.name_scope(scope):
        return tf.random_normal(tf.shape(mean), mean, tf.sqrt(var))

def duplicate(x, n_iw=1, n_mc=1, scope=None):
    """ Duplication function adds samples according to n_iw and n_mc.

    This function is specifically for importance weighting and monte carlo
    sampling.
    """
    with tf.name_scope(scope):
        sample_shape = x._shape_as_list()[1:]
        y = tf.reshape(x, [1, 1, -1] + sample_shape)
        y = tf.tile(y, [n_iw, n_mc, 1] + [1] * len(sample_shape))
        y = tf.reshape(y, [-1] + sample_shape)
    return y

GaussianSample = gaussian_sample
Duplicate = duplicate

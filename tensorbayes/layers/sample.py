import tensorflow as tf

def gaussian_sample(mean, var, scope=None):
    with tf.variable_scope(scope, 'gaussian_sample'):
        sample = tf.random_normal(tf.shape(mean), mean, tf.sqrt(var))
        sample.set_shape(mean.get_shape())
        return sample

def bernoulli_sample(mean, scope=None):
    with tf.variable_scope(scope, 'bernoulli_sample'):
        sample = tf.cast(
            tf.greater(mean, tf.random_uniform(tf.shape(mean), 0, 1)),
            tf.float32)
        sample.set_shape(mean.get_shape())
        return sample

def duplicate(x, n_iw=1, n_mc=1, scope=None):
    """ Duplication function adds samples according to n_iw and n_mc.

    This function is specifically for importance weighting and monte carlo
    sampling.
    """
    with tf.variable_scope(scope, 'duplicate'):
        sample_shape = x._shape_as_list()[1:]
        y = tf.reshape(x, [1, 1, -1] + sample_shape)
        multiplier = tf.stack([n_iw, n_mc, 1] + [1] * len(sample_shape))
        y = tf.tile(y, multiplier)
        y = tf.reshape(y, [-1] + sample_shape)
    return y

GaussianSample = gaussian_sample
BernoulliSample = bernoulli_sample
Duplicate = duplicate

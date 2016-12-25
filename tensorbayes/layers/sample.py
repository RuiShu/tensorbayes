import tensorflow as tf

def GaussianSample(mean, var, scope):
    with tf.name_scope(scope):
        return tf.random_normal(tf.shape(mean), mean, tf.sqrt(var))

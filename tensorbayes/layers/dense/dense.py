import tensorflow as tf


def dense(x, size, scope=None, activation=None, reuse=None):
    return tf.contrib.layers.fully_connected(
        x, size, scope=scope, activation_fn=activation, reuse=reuse
    )

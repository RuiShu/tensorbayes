import tensorflow as tf


def placeholder(shape, dtype=tf.float32, name=None):
    return tf.placeholder(dtype, shape, name=name)

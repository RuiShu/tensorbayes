import tensorflow as tf


def constant(value, name=None):
    return tf.constant(value, tf.float32, name=name)

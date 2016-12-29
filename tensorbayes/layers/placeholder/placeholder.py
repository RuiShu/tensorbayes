import tensorflow as tf


class Placeholder(object):
    """TensorBayes Placeholder Class

    Wraps functionality of a TensorFlow placeholder.
    """
    def __init__(self, shape, dtype=tf.float32, name=None):
        self.shape = shape
        self.dtype = dtype
        self.name = name

    def __call__(self):
        return tf.placeholder(self.dtype, self.shape, name=self.name)

import tensorflow as tf


class Constant(object):
    """TensorBayes Constant Class

    Wraps functionality of a TensorFlow constant.
    """
    def __init__(self, value, dtype=tf.float32, name=None):
        self.value = value
        self.dtype = dtype
        self.name = name

    def __call__(self):
        return tf.constant(self.value, self.dtype, self.name)

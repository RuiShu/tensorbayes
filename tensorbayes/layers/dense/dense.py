import tensorflow as tf


class Dense(object):
    """TensorBayes Dense Layer"""
    def __init__(self, x, size, scope, activation=None, reuse=None):
        self.x = x
        self.size = size
        self.scope = scope
        self.activation = activation
        self.reuse = reuse

    def __call__(self):
        return tf.contrib.layers.fully_connected(
            self.x, self.size, scope=self.scope, activation_fn=self.activation,
            reuse=self.reuse
        )

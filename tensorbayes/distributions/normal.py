import tensorflow as tf
import numpy as np


class Normal(object):
    """TensorBayes Normal Univariate Distribution Class"""
    def __init__(self, mu, var):
        self.mu = mu
        self.var = var

    def log_pdf(self, x):
        mu, var = self.mu, self.var
        return -0.5 * tf.reduce_sum(
            tf.log(2 * np.pi) + tf.log(var) + tf.square(x - mu) / var, 1
        )

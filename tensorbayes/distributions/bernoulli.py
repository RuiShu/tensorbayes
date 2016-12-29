import tensorflow as tf


class Bernoulli(object):
    """TensorBayes Bernoulli Distribution Class"""
    def __init__(self, logits):
        self.logits = logits

    def log_pdf(self, x):
        logits = self.logits
        return -tf.reduce_sum(
            tf.nn.sigmoid_cross_entropy_with_logits(logits, x), 1
        )

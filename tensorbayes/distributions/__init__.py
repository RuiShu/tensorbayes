import numpy as np
import tensorflow as tf

def normal_log_pdf(x, mu, var):
    """Normal log-density."""
    return -0.5 * tf.reduce_sum(
        tf.log(2 * np.pi) + tf.log(var) + tf.square(x - mu) / var, 1
    )

def bernoulli_log_pdf(x, logits):
    """Bernoulli log-density."""
    return -tf.reduce_sum(
        tf.nn.sigmoid_cross_entropy_with_logits(logits, x), 1
    )


""" Assumes softplus activations for gaussian
"""
import tensorflow as tf
import numpy as np

def log_bernoulli_with_logits(x, logits, eps=0.0, axis=-1):
    if eps > 0.0:
        max_val = np.log(1.0 - eps) - np.log(eps)
        logits = tf.clip_by_value(logits, -max_val, max_val,
                                  name='clipped_logit')
    # return -tf.reduce_sum(
    #     tf.nn.sigmoid_cross_entropy_with_logits(logits, x), axis)
    return -tf.reduce_sum(
        tf.nn.sigmoid_cross_entropy_with_logits(logits=logits, labels=x), axis)

def log_normal(x, mu, var, eps=0.0, axis=-1):
    if eps > 0.0:
        var = tf.add(var, eps, name='clipped_var')
    return -0.5 * tf.reduce_sum(
        tf.log(2 * np.pi) + tf.log(var) + tf.square(x - mu) / var, axis)

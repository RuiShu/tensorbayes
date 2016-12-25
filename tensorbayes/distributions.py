""" Assumes softplus activations for gaussian
"""
import tensorflow as tf
import numpy as np

def log_bernoulli_with_logits(x, logits):
    return -tf.reduce_sum(tf.nn.sigmoid_cross_entropy_with_logits(logits, x), 1)

def log_normal(x, mu, var):
    return -0.5 * tf.reduce_sum(tf.log(2 * np.pi) + tf.log(var) + tf.square(x - mu) / var, 1)

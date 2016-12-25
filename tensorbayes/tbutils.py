import tensorflow as tf

def log_sum_exp(x, axis=1, keep_dims=False):
    a = tf.reduce_max(x, axis, keep_dims=True)
    out = a + tf.log(tf.reduce_sum(tf.exp(x - a), axis, keep_dims=True))
    if keep_dims:
        return out
    else:
        return tf.squeeze(out, [axis])

def cross_entropy_with_logits(logits, targets):
    log_q = tf.nn.log_softmax(logits)
    return -tf.reduce_sum(targets * log_q, 1)

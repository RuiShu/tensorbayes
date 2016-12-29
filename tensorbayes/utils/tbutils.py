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

def ones_initializer(shape, dtype=tf.float32, partition_info=None):
    return tf.constant(1, dtype=dtype, shape=shape)

def zeros_initializer(shape, dtype=tf.float32, partition_info=None):
    return tf.constant(0, dtype=dtype, shape=shape)

def assign_moving_average(orig_val, new_val, decay, name):
    with tf.name_scope(name):
        td = decay * (new_val - orig_val)
        return tf.assign_add(orig_val, td)

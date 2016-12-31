import tensorflow as tf


def batch_normalization(x, phase, scope=None, reuse=None):
    """Batch normalization for TensorFlow r0.9
    """
    with tf.name_scope(scope):
        b_tr = lambda: tf.contrib.layers.batch_norm(
            x, center=True, scale=True, is_training=True, reuse=reuse,
            scope=scope
        )
        b_t = lambda: tf.contrib.layers.batch_norm(
            x, center=True, scale=True, is_training=False, reuse=True,
            scope=scope
        )
        bn = tf.cond(phase, b_tr, b_t, name=scope)

    return bn

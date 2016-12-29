import tensorflow as tf


class BatchNormalization(object):
    """TensorBayes Batch Normalization Layer"""
    def __init__(self, x, phase, scope=None, reuse=None):
        self.x = x
        self.phase = phase
        self.scope = scope
        self.reuse = reuse

    def __call__(self):
        # Extract variables for subsequent use.
        x = self.x
        phase = self.phase
        scope = self.scope
        reuse = self.reuse

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


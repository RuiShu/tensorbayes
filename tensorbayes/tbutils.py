import tensorflow as tf

def log_sum_exp(x, axis=1, keep_dims=False):
    a = tf.reduce_max(x, axis, keep_dims=True)
    out = a + tf.log(tf.reduce_sum(tf.exp(x - a), axis, keep_dims=True))
    if keep_dims:
        return out
    else:
        if type(axis) is list:
            return tf.squeeze(out, axis)
        else:
            return tf.squeeze(out, [axis])

def cross_entropy_with_logits(logits, targets):
    """
    TODO: make this fn numerically stable.
    """
    log_q = tf.nn.log_softmax(logits)
    return -tf.reduce_sum(targets * log_q, 1)

def clip_gradients(optimizer, loss, max_clip=0.9, max_norm=4):
    grads_and_vars = optimizer.compute_gradients(loss)
    # Filter for non-None grads
    grads_and_vars = [gv for gv in grads_and_vars if gv[0] is not None]
    grads = [g for g, _ in grads_and_vars]
    grads, global_grad_norm = tf.clip_by_global_norm(grads, max_norm)
    clipped_grads_and_vars = []
    for i in xrange(len(grads_and_vars)):
        g = tf.clip_by_value(grads[i], -max_clip, max_clip)
        v = grads_and_vars[i][1]
        clipped_grads_and_vars += [(g, v)]
    return clipped_grads_and_vars, global_grad_norm

class Function(object):
    def __init__(self, sess, inputs, outputs):
        self.inputs = inputs
        self.outputs = outputs
        self.sess = sess

    def __call__(self, *args):
        feeds = {}
        for (i, arg) in enumerate(args):
            feeds[self.inputs[i]] = arg
        return self.sess.run(self.outputs, feeds)

def function(sess, inputs, outputs):
    return Function(sess, inputs, outputs)

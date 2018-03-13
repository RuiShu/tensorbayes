import tensorflow as tf
from tensorflow.python.ops.nn_ops import softmax_cross_entropy_with_logits_v2 as softmax_xent

def accuracy(x, y, scope=None):
    with tf.variable_scope(scope, 'acc') as sc:
        x = tf.argmax(x, 1)
        y = tf.argmax(y, 1)
        _, acc = tf.metrics.accuracy(x, y)
        acc_init = tf.variables_initializer(tf.get_collection('local_variables', sc.name))
    return acc, acc_init

def reduce_sum_sq(x, axis=None, keepdims=False, name=None):
    with tf.name_scope(name):
        return tf.reduce_sum(tf.square(x), axis=axis, keepdims=keepdims)

def reduce_l2_loss(x, axis=None, keepdims=False, name=None):
    with tf.name_scope(name):
        return reduce_sum_sq(x, axis=axis, keepdims=keepdims) / 2

def log_sum_exp(x, axis=1, keepdims=False):
    a = tf.reduce_max(x, axis, keepdims=True)
    out = a + tf.log(tf.reduce_sum(tf.exp(x - a), axis, keepdims=True))
    if keepdims:
        return out
    else:
        if type(axis) is list:
            return tf.squeeze(out, axis)
        else:
            return tf.squeeze(out, [axis])

def softmax_cross_entropy_with_two_logits(logits=None, labels=None):
    return softmax_xent(labels=tf.nn.softmax(labels), logits=logits)

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

class TensorDict(object):
    def __init__(self, d={}):
        self.__dict__ = dict(d)

    def __iter__(self):
        return iter(self.__dict__)

    def __contains__(self, key):
        return key in self.__dict__

    def __getitem__(self, key):
        return self.__dict__[key]

    def __repr__(self):
        return repr(self.__dict__)

def growth_config(*args, **kwargs):
    config = tf.ConfigProto(*args, **kwargs)
    config.gpu_options.allow_growth = True
    return config

def get_getter(ema):
    def ema_getter(getter, name, *args, **kwargs):
        var = getter(name, *args, **kwargs)
        ema_var = ema.average(var)
        return ema_var if ema_var else var
    return ema_getter

import tensorflow as tf
from tensorflow.contrib.framework import add_arg_scope, arg_scope

def _assign_moving_average(orig_val, new_val, momentum, name):
    with tf.name_scope(name):
        scaled_diff = (1 - momentum) * (new_val - orig_val)
        return tf.assign_add(orig_val, scaled_diff)

@add_arg_scope
def batch_norm(x,
               phase,
               shift=True,
               scale=True,
               momentum=0.99,
               eps=1e-3,
               internal_update=False,
               scope=None,
               reuse=None):

    C = x._shape_as_list()[-1]
    ndim = len(x.shape)
    var_shape = [1] * (ndim - 1) + [C]

    with tf.variable_scope(scope, 'batch_norm', reuse=reuse):
        def training():
            m, v = tf.nn.moments(x, range(ndim - 1), keep_dims=True)
            update_m = _assign_moving_average(moving_m, m, momentum, 'update_mean')
            update_v = _assign_moving_average(moving_v, v, momentum, 'update_var')
            tf.add_to_collection('update_ops', update_m)
            tf.add_to_collection('update_ops', update_v)

            if internal_update:
                with tf.control_dependencies([update_m, update_v]):
                    output = (x - m) * tf.rsqrt(v + eps)
            else:
                output = (x - m) * tf.rsqrt(v + eps)
            return output

        def testing():
            m, v = moving_m, moving_v
            output = (x - m) * tf.rsqrt(v + eps)
            return output

        # Get mean and variance, normalize input
        moving_m = tf.get_variable('mean', var_shape, initializer=tf.zeros_initializer, trainable=False)
        moving_v = tf.get_variable('var', var_shape, initializer=tf.ones_initializer, trainable=False)

        if isinstance(phase, bool):
            output = training() if phase else testing()
        else:
            output = tf.cond(phase, training, testing)

        if scale:
            output *= tf.get_variable('gamma', var_shape, initializer=tf.ones_initializer)

        if shift:
            output += tf.get_variable('beta', var_shape, initializer=tf.zeros_initializer)

    return output

@add_arg_scope
def instance_norm(x,
                  shift=True,
                  scale=True,
                  eps=1e-3,
                  scope=None,
                  reuse=None):

    # Expect a 4-D Tensor
    C = x._shape_as_list()[-1]

    with tf.variable_scope(scope, 'instance_norm', reuse=reuse):
        # Get mean and variance, normalize input
        m, v = tf.nn.moments(x, [1, 2], keep_dims=True)
        output = (x - m) * tf.rsqrt(v + eps)

        if scale:
            output *= tf.get_variable('gamma', C, initializer=tf.ones_initializer)

        if shift:
            output += tf.get_variable('beta', C, initializer=tf.zeros_initializer)

    return output

@add_arg_scope
def context_shift(x,
                  context,
                  shift=True,
                  scale=True,
                  scope=None,
                  reuse=None):

    B = context._shape_as_list()[-1]
    C = x._shape_as_list()[-1]
    ndim = len(x.shape)
    var_shape = [B] + [1] * (ndim - 2) + [C]

    with tf.variable_scope(scope, 'context_shift', reuse=reuse):
        output = x

        if scale:
            gamma = tf.get_variable('gamma', var_shape, initializer=tf.ones_initializer)
            output *= tf.tensordot(context, gamma, 1)

        if shift:
            beta = tf.get_variable('beta', var_shape, initializer=tf.zeros_initializer)
            output += tf.tensordot(context, beta, 1)

        output.set_shape(x.get_shape())

    return output

@add_arg_scope
def lookup_shift(x,
                 context,
                 shift=True,
                 scale=True,
                 scope=None,
                 reuse=None):

    B = context._shape_as_list()[-1]
    C = x._shape_as_list()[-1]
    ndim = len(x.shape)
    var_shape = [B] + [1] * (ndim - 2) + [C]

    with tf.variable_scope(scope, 'lookup_shift', reuse=reuse):
        output = x
        ids = tf.argmax(context, -1)

        if scale:
            gamma = tf.get_variable('gamma', var_shape, initializer=tf.ones_initializer)
            output *= tf.nn.embedding_lookup(gamma, ids)

        if shift:
            beta = tf.get_variable('beta', var_shape, initializer=tf.zeros_initializer)
            output += tf.nn.embedding_lookup(beta, ids)

    return output

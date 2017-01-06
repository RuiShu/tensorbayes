import tensorflow as tf

def batch_norm(x, phase, scope=None, reuse=None):
    return tf.contrib.layers.batch_norm(x, center=True, scale=True,
                                        decay=0.99,
                                        is_training=phase,
                                        reuse=reuse, scope='bn')

def _ones_initializer(shape, dtype=tf.float32, partition_info=None):
    return tf.constant(1, dtype=dtype, shape=shape)

def _zeros_initializer(shape, dtype=tf.float32, partition_info=None):
    return tf.constant(0, dtype=dtype, shape=shape)

def _assign_moving_average(orig_val, new_val, momentum, name):
    with tf.name_scope(name):
        scaled_diff = (1 - momentum) * (new_val - orig_val)
        return tf.assign_add(orig_val, scaled_diff)

def custom_batch_norm(x, train, eps=1e-3, momentum=0.99, scope=None,
                      reuse=None, reuse_averages=True):
    var_shape = tf.TensorShape(x.get_shape()[-1])
    with tf.variable_scope(scope, 'BatchNorm') as sc:
        if not reuse_averages:
            if tf.get_variable_scope().reuse:
                raise Exception("Cannot create stream-unique averages if "
                                "reuse flag is ON")
            i = len(tf.get_collection(tf.GraphKeys.VARIABLES,
                                      scope='{:s}/moving_mean'.format(sc.name)))
            mm = tf.get_variable('moving_mean/v{:d}'.format(i), var_shape,
                                 initializer=_zeros_initializer)
            mv = tf.get_variable('moving_variance/v{:d}'.format(i), var_shape,
                                 initializer=_ones_initializer)
        else:
            if reuse: sc.reuse_variables()
            mm = tf.get_variable('moving_mean/v0', var_shape,
                                 initializer=_zeros_initializer)
            mv = tf.get_variable('moving_variance/v0', var_shape,
                                 initializer=_ones_initializer)
        if reuse: sc.reuse_variables()
        m, v = tf.nn.moments(x, range(len(x.get_shape()) - 1), keep_dims=False)
        beta = tf.get_variable('beta', x.get_shape()[-1],
                               initializer=_zeros_initializer)
        gamm = tf.get_variable('gamm', x.get_shape()[-1],
                               initializer=_ones_initializer)
        def training():
            with tf.name_scope('training'):
                with tf.name_scope('update_moments'):
                    update_m = _assign_moving_average(mm, m, momentum,
                                                      name='update_mean')
                    update_v = _assign_moving_average(mv, v, momentum,
                                                      name='update_var')
                    tf.add_to_collection(tf.GraphKeys.UPDATE_OPS, update_m)
                    tf.add_to_collection(tf.GraphKeys.UPDATE_OPS, update_v)
                return tf.nn.batch_normalization(x, m, v, beta, gamm, eps,
                                                 name='batch_normalized')
        def testing():
            with tf.name_scope('testing'):
                return tf.nn.batch_normalization(x, mm, mv, beta, gamm, eps,
                                                 name='pop_normalized')
        return tf.cond(train, training, testing, name='batch_pop_switch')

BatchNormalization = batch_norm
CustomBatchNormalization = custom_batch_norm

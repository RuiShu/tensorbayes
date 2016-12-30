import tensorflow as tf

def Constant(value, name=None):
    return tf.constant(value, 'float32', name=name)

def Placeholder(shape, dtype='float32', name=None):
    return tf.placeholder(dtype, shape, name=name)

def Dense(x, size, scope=None, activation=None, reuse=None):
    return tf.contrib.layers.fully_connected(x, size, scope=scope,
                                             activation_fn=activation,
                                             reuse=reuse)

def GaussianUpdate(zm1, zv1, zm2, zv2, scope=None):
    with tf.name_scope(scope):
        with tf.name_scope('variance'):
            zp1 = 1.0/zv1
            zp2 = 1.0/zv2
            zv = 1.0/(zp1 + zp2)
        with tf.name_scope('mean'):
            zm = (zm1 * zp1 + zm2 * zp2) * zv
    return zm, zv

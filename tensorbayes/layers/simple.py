import tensorflow as tf

def Constant(value, name=None):
    return tf.constant(value, 'float32', name=name)

def Placeholder(shape, dtype='float32', name=None):
    return tf.placeholder(dtype, shape, name=name)

def Dense(x, size, scope=None, activation=None, reuse=None):
    return tf.contrib.layers.fully_connected(x, size, scope=scope,
                                             activation_fn=activation,
                                             reuse=reuse)

def GaussianUpdate(zm1, zv1, zm2, zv2, scope=None, eps=0.0):
    with tf.name_scope(scope):
        with tf.name_scope('variance'):
            if eps > 0.0:
                """It is not clear to me yet whether this will cause our loss function to be
                severely biased
                """
                raise Exception("Adding eps noise deprecated at the moment for GaussianUpdate fn")
                zv1 = tf.add(zv1, eps, name='clip_var1')
                zv2 = tf.add(zv2, eps, name='clip_var2')
            zp1 = 1.0/zv1
            zp2 = 1.0/zv2
            zv = 1.0/(zp1 + zp2)
        with tf.name_scope('mean'):
            zm = (zm1 * zp1 + zm2 * zp2) * zv
    return zm, zv

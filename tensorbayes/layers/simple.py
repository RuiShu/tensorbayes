import tensorflow as tf

def Constant(value, name=None):
    return tf.constant(value, 'float32', name=name)

def Placeholder(shape, name):
    return tf.placeholder('float32', shape, name=name)

def Dense(x, size, scope, activation=None, reuse=None):
    return tf.contrib.layers.fully_connected(x, size, scope=scope,
                                             activation_fn=activation,
                                             reuse=reuse)

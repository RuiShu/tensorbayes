import tensorflow as tf
from tensorflow.contrib.framework import add_arg_scope
from .normalization import *
from tensorflow.contrib.layers import xavier_initializer
import numpy as np

def constant(value, name=None):
    return tf.constant(value, 'float32', name=name)

def placeholder(shape, dtype='float32', name=None):
    return tf.placeholder(dtype, shape, name=name)

@add_arg_scope
def dense(x,
          num_outputs,
          scope=None,
          activation=None,
          reuse=None,
          bn=False,
          phase=None):

    with tf.variable_scope(scope, 'dense', reuse=reuse):
        # convert x to 2-D tensor
        dim = np.prod(x._shape_as_list()[1:])
        x = tf.reshape(x, [-1, dim])
        weights_shape = (x.get_shape().dims[-1], num_outputs)

        # dense layer
        weights = tf.get_variable('weights', weights_shape,
                                  initializer=xavier_initializer())
        biases = tf.get_variable('biases', [num_outputs],
                                 initializer=tf.zeros_initializer)
        output = tf.matmul(x, weights) + biases
        if bn: output = batch_norm(output, phase, scope='bn')
        if activation: output = activation(output)
    return output

@add_arg_scope
def conv2d(x,
           num_outputs,
           kernel_size,
           strides,
           padding='SAME',
           activation=None,
           bn=False,
           phase=None,
           scope=None,
           reuse=None):
    # Convert int to list
    kernel_size = [kernel_size] * 2 if isinstance(kernel_size, int) else kernel_size
    strides = [strides] * 2 if isinstance(strides, int) else strides

    # Convert list to valid list
    kernel_size = list(kernel_size) + [x.get_shape().dims[-1], num_outputs]
    strides = [1] + list(strides) + [1]

    # Conv operation
    with tf.variable_scope(scope, 'conv2d', reuse=reuse):
        kernel = tf.get_variable('weights', kernel_size,
                                 initializer=xavier_initializer())
        biases = tf.get_variable('biases', [num_outputs],
                                 initializer=tf.zeros_initializer)
        output = tf.nn.conv2d(x, kernel, strides, padding, name='conv2d')
        output += biases
        if bn: output = batch_norm(output, phase, scope='bn')
        if activation: output = activation(output)
    return output

@add_arg_scope
def conv2d_transpose(x,
                     num_outputs,
                     kernel_size,
                     strides,
                     padding='SAME',
                     output_shape=None,
                     output_like=None,
                     activation=None,
                     bn=False,
                     phase=None,
                     scope=None,
                     reuse=None):
    # Get output shape both as tensor obj and as list
    if output_like is None:
        bs = tf.shape(x)[0]
        _output_shape = tf.pack([bs] + output_shape[1:])
    else:
        _output_shape = tf.shape(output_like)
        output_shape = output_like.get_shape()

    # Convert int to list
    kernel_size = [kernel_size] * 2 if isinstance(kernel_size, int) else kernel_size
    strides = [strides] * 2 if isinstance(strides, int) else strides

    # Convert list to valid list
    kernel_size = list(kernel_size) + [num_outputs, x.get_shape().dims[-1]]
    strides = [1] + list(strides) + [1]

    # Transposed conv operation
    with tf.variable_scope(scope, 'conv2d', reuse=reuse):
        kernel = tf.get_variable('weights', kernel_size,
                                 initializer=xavier_initializer())
        biases = tf.get_variable('biases', [num_outputs],
                                 initializer=tf.zeros_initializer)
        output = tf.nn.conv2d_transpose(x, kernel, _output_shape, strides,
                                        padding, name='conv2d_transpose')
        output += biases
        output.set_shape(output_shape)
        if bn: output = batch_norm(output, phase, scope='bn')
        if activation: output = activation(output)
    return output

@add_arg_scope
def upsample(x,
             strides,
             scope=None):
    # Convert int to list
    strides = [strides] * 2 if isinstance(strides, int) else strides
    shape = x._shape_as_list()[1:3]
    h, w = strides[0] * shape[0], strides[1] * shape[1]

    with tf.variable_scope(scope, 'upsample'):
        output = tf.image.resize_nearest_neighbor(x, [h, w])
    return output

@add_arg_scope
def max_pool(x,
             kernel_size,
             strides,
             padding='SAME',
             scope=None):
    # Convert int to list
    kernel_size = [kernel_size] * 2 if isinstance(kernel_size, int) else kernel_size
    strides = [strides] * 2 if isinstance(strides, int) else strides

    # Convert list to valid list
    kernel_size = [1] + list(kernel_size) + [1]
    strides = [1] + list(strides) + [1]

    with tf.variable_scope(scope, 'max_pool'):
        output = tf.nn.max_pool(x, kernel_size, strides, padding=padding)
    return output

@add_arg_scope
def avg_pool(x,
             kernel_size,
             strides,
             padding='SAME',
             scope=None):
    # Convert int to list
    kernel_size = [kernel_size] * 2 if isinstance(kernel_size, int) else kernel_size
    strides = [strides] * 2 if isinstance(strides, int) else strides

    # Convert list to valid list
    kernel_size = [1] + list(kernel_size) + [1]
    strides = [1] + list(strides) + [1]

    with tf.variable_scope(scope, 'avg_pool'):
        output = tf.nn.avg_pool(x, kernel_size, strides, padding=padding)
    return output

@add_arg_scope
def gaussian_update(zm1, zv1,
                    zm2, zv2,
                    scope=None,
                    eps=0.0):
    with tf.variable_scope(scope, 'gaussian_update'):
        with tf.name_scope('variance'):
            if eps > 0.0:
                """It is not clear to me yet whether this will cause our loss
                function to be severely biased
                """
                raise Exception("Adding eps noise deprecated at the moment "
                                "for gaussian update fn")
                zv1 = tf.add(zv1, eps, name='clip_var1')
                zv2 = tf.add(zv2, eps, name='clip_var2')
            zp1 = 1.0/zv1
            zp2 = 1.0/zv2
            zv = 1.0/(zp1 + zp2)
        with tf.name_scope('mean'):
            zm = (zm1 * zp1 + zm2 * zp2) * zv
    return zm, zv

Constant = constant
Placeholder = placeholder
Dense = dense
GaussianUpdate = gaussian_update

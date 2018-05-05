"""
Basic wall-clock test for a generic convolutional neural network

Tesla K40c:
Elapsed wall-clock time: 58.0986320972
Average time per iter: 0.0580986320972

GeForce GTX 1080 Ti:
Elapsed wall-clock time: 41.549612999
Average time per iter: 0.041549612999

GeForce GTX TITAN X:
Elapsed wall-clock time: 61.8042290211
Average time per iter: 0.0618042290211
"""

import numpy as np
import tensorflow as tf
import tensorbayes as tb
from tensorflow.contrib.framework import arg_scope
from tensorbayes.layers import conv2d, dense
from tensorflow.python.ops.nn_ops import softmax_cross_entropy_with_logits_v2 as softmax_xent
import time
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

def classifier(x, phase, reuse=None):
    with tf.variable_scope('class', reuse=reuse):
        with arg_scope([conv2d, dense], bn=True, phase=phase, activation=tf.nn.relu):
            for i in range(4):
                x = conv2d(x, 64 + 64 * i, 3, 2)
                x = conv2d(x, 64 + 64 * i, 3, 1)

            x = dense(x, 500)
            x = dense(x, 10, activation=None)

    return x

def build_graph():
    T = tb.TensorDict(dict(
        sess = tf.Session(config=tb.growth_config()),
        x = tb.nn.placeholder((None, 32, 32, 3)),
        y = tb.nn.placeholder((None, 10)),
    ))

    y = classifier(T.x, phase=True)
    loss = tf.reduce_mean(softmax_xent(labels=T.y, logits=y))
    train_main = tf.train.AdamOptimizer().minimize(loss)
    T.main_ops = [train_main, loss]
    T.sess.run(tf.global_variables_initializer())
    return T

def train(T):
    for i in range(1000):
        x = np.random.randn(100, 32, 32, 3)
        y = np.tile(np.eye(10)[0], (100, 1))
        _, loss = T.sess.run(T.main_ops, feed_dict={T.x: x, T.y: y})
        tb.utils.progbar(i, 1000, '{} / {:.2f}'.format(i, loss))


if __name__ == '__main__':
    T = build_graph()
    t = time.time()
    train(T)
    t = time.time() - t
    print('Elapsed wall-clock time: {}'.format(t))
    print('Average time per iter: {}'.format(t / 1000))

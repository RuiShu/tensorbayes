import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import numpy as np
import sys

mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

def progbar(i, iter_per_epoch):
    j = (i % iterep) + 1
    perc = int(100. * j / iter_per_epoch)
    prog = ''.join(['='] * (perc/2))
    string = "\r[{:50s}] {:3d}%".format(prog, perc)
    sys.stdout.write(string); sys.stdout.flush()
    if j == iter_per_epoch:
        sys.stdout.write('\r{:100s}\r'.format('')); sys.stdout.flush()

# Layers for nn
def Constant(value, name=None):
    return tf.constant(value, 'float32', name=name)

def Dense(x, size, scope, activation=None, reuse=None):
    return tf.contrib.layers.fully_connected(x, size, scope=scope, activation_fn=activation, reuse=reuse)

def Sample(mean, var, scope):
    with tf.name_scope(scope):
        return tf.random_normal(tf.shape(mean), mean, tf.sqrt(var))

# probability density functions
def log_bernoulli_with_logits(x, logits):
    return -tf.reduce_sum(tf.nn.sigmoid_cross_entropy_with_logits(logits, x), 1)

def log_normal(x, mu, var):
    return -0.5 * tf.reduce_sum(tf.log(2 * np.pi) + tf.log(var) + tf.square(x - mu) / var, 1)

# ssl-vae subgraphs
def alpha_graph(x, reuse=None):
    # -- q(y)
    with tf.variable_scope('qy'):
        h1 = Dense(x, 100, 'layer1', tf.nn.relu, reuse=reuse)
        h2 = Dense(h1, 100, 'layer2', tf.nn.relu, reuse=reuse)
        qy_logit = Dense(h2, 10, 'logit', reuse=reuse)
        qy = tf.nn.softmax(qy_logit, name='prob')
    return qy_logit, qy

def xy_graph(x, y, reuse=None):
    # -- q(z)
    with tf.variable_scope('qz'):
        xy = tf.concat(1, (x, y), name='xy/concat')
        h1 = Dense(xy, 500, 'layer1', tf.nn.relu, reuse=reuse)
        h2 = Dense(h1, 500, 'layer2', tf.nn.relu, reuse=reuse)
        zm = Dense(h2, 50, 'zm', reuse=reuse)
        zv = Dense(h2, 50, 'zv', tf.nn.softplus, reuse=reuse)
        z = Sample(zm, zv, 'z')
    # -- p(x)
    with tf.variable_scope('px'):
        zy = tf.concat(1, (z, y), name='zy/concat')
        h1 = Dense(zy, 500, 'layer1', tf.nn.relu, reuse=reuse)
        h2 = Dense(h1, 500, 'layer2', tf.nn.relu, reuse=reuse)
        px_logit = Dense(h2, 784, 'logit', reuse=reuse)
    return zm, zv, z, px_logit

# ssl-vae loss
def labeled_loss(x, px_logit, z, zm, zv):
    xy_loss = -log_bernoulli_with_logits(x, px_logit)
    xy_loss += log_normal(z, zm, zv) - log_normal(z, tf.constant(0, 'float32'), tf.constant(1, 'float32'))
    return xy_loss

tf.reset_default_graph()
with tf.name_scope('l'):
    x = tf.placeholder('float32', (None, 784), name='x')
    y = tf.placeholder('float32', (None, 10), name='y')
    qy_logit, qy = alpha_graph(x)
    with tf.name_scope('graph'):
        zm, zv, z, px_logit = xy_graph(x, y)

with tf.name_scope('accuracy'):
    accuracy = tf.reduce_mean(tf.cast(
        tf.equal(tf.argmax(y, 1), tf.argmax(qy_logit, 1)), 'float32'))

with tf.name_scope('alpha_loss'):
    alpha = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(qy_logit, y))

with tf.name_scope('xy_loss'):
    xy_loss = tf.reduce_mean(labeled_loss(x, px_logit, z, zm, zv))

with tf.name_scope('u'):
    x = tf.placeholder('float32', (None, 784), name='x')
    with tf.name_scope('y_'):
        y_ = tf.fill(tf.pack([tf.shape(x)[0], 10]), 0.0)
    qy_logit, qy = alpha_graph(x, reuse=True)

    zm, zv, z, px_logit = [None] * 10, [None] * 10, [None ] * 10, [None] * 10
    for i in xrange(10):
        with tf.name_scope('graph{:d}'.format(i)):
            y = tf.add(y_, tf.constant(np.eye(10)[i], 'float32', name='hot_at_{:d}'.format(i)))
            zm[i], zv[i], z[i], px_logit[i] = xy_graph(x, y, reuse=True)

with tf.name_scope('x_loss'):
    x_loss = -tf.nn.softmax_cross_entropy_with_logits(qy_logit, qy)
    for i in xrange(10):
        val = labeled_loss(x, px_logit[i], z[i], zm[i], zv[i])
        x_loss += qy[:, i] * val
    x_loss = tf.reduce_mean(x_loss)

with tf.name_scope('loss'):
    loss = alpha + xy_loss + x_loss
    # loss = alpha * 0.1 + xy_loss * 100./50000. + x_loss

train_step = tf.train.AdamOptimizer().minimize(loss)
sess = tf.Session()

# Change initialization protocol depending on tensorflow version
sess.run(tf.initialize_all_variables())
# sess.run(tf.global_variables_initializer())
label_xs, label_ys = mnist.train.next_batch(100)

iterep = 500
for i in range(iterep * 1000):
    unlab_xs, _ = mnist.train.next_batch(100)
    _, a, b, c = sess.run([train_step, alpha, xy_loss, x_loss],
             feed_dict={'l/x:0': label_xs,
                        'l/y:0': label_ys,
                        'u/x:0': unlab_xs})
    progbar(i, iterep)
    if (i + 1) %  iterep == 0:
        tr = sess.run(accuracy, feed_dict={'l/x:0': label_xs, 'l/y:0': label_ys})
        te = sess.run(accuracy, feed_dict={'l/x:0': mnist.test.images, 'l/y:0': mnist.test.labels})
        string = '\t{:>8s},\t{:>8s},\t{:>8s},\t{:>8s},\t{:>8s},\t{:>8s}'
        print string.format('alpha', 'xy_loss', 'x_loss', 'tr_acc', 'te_acc', 'epoch')
        string = '\t{:8.2e},\t{:8.2f},\t{:8.2f},\t{:8.2f},\t{:8.2f},\t{:8d}'
        print string.format(a, b, c, tr * 100, te * 100, (i + 1) / iterep)

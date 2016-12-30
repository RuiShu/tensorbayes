import argparse
import numpy as np
import tensorflow as tf
import tensorbayes as tb
from tensorbayes.layers import Constant, Placeholder
from tensorbayes.layers import Dense, BatchNormalization, GaussianUpdate
from tensorbayes.layers import GaussianSample
from tensorbayes.nbutils import show_graph
from tensorbayes.utils import progbar
from tensorbayes.distributions import log_bernoulli_with_logits
from tensorbayes.distributions import log_normal
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

# General settings and modifications
parser = argparse.ArgumentParser()
parser.add_argument("-run", type=int, help="Run index")
parser.add_argument("-bs", type=int, help="Minibatch size", default=256)
parser.add_argument("-lr", type=float, help="learning rate", default=2e-4)
parser.add_argument("-nonlin", type=str, help="activation function", default='elu')
parser.add_argument("-eps", type=float, help="distribution epsilon", default=1e-5)
args = parser.parse_args()
if any([k[1] is None for k in args._get_kwargs()]):
    print [k[1] for k in args._get_kwargs()]
    parser.error("Some arguments not provided.")
else:
    print "Command line arguments"
    print args
if args.nonlin == 'relu':
    activate = tf.nn.relu
elif args.nonlin == 'elu':
    activate = tf.nn.elu
else:
    raise Exception('Unexpected nonlinearity arg')
log_bern = lambda x, logits: log_bernoulli_with_logits(x, logits, args.eps)
log_norm = lambda x, mu, var: log_normal(x, mu, var, args.eps)

# Convenience layers and graph blocks
def name(index, suffix):
    return 'z{:d}'.format(index) + '_' + suffix

def layer(x, size, scope, bn=True, activation=None):
    with tf.variable_scope(scope):
        h = Dense(x, size, scope='dense')
        if bn: h = BatchNormalization(h, phase, scope='bn')
        if activation is not None: h = activation(h)
        return h

def encode_block(x, h_size, z_size, idx):
    with tf.variable_scope(name(idx, 'encode')):
        h = layer(x, h_size, 'layer1', activation=activate)
        h = layer(h, h_size, 'layer2', activation=activate)
    with tf.variable_scope(name(idx, 'encode/likelihood')):
        z_m = layer(h, z_size, 'mean')
        z_v = layer(h, z_size, 'variance', activation=tf.nn.softplus)
    return (z_m, z_v)

def infer_block(likelihood, prior, idx):
    with tf.variable_scope(name(idx, 'sample')):
        if prior is None:
            posterior = likelihood
        else:
            args = likelihood + prior
            posterior = GaussianUpdate(*args, scope='pwm')
        z = GaussianSample(*posterior, scope='sample')
    return z, posterior

def decode_block(z_like, z_prior, h_size, x_size, idx):
    z, z_post = infer_block(z_like, z_prior, idx)
    with tf.variable_scope(name(idx - 1, 'decode')):
        h = layer(z, h_size, 'layer1', activation=activate)
        h = layer(h, h_size, 'layer2', activation=activate)
    with tf.variable_scope(name(idx - 1, 'decode/prior')):
        if (idx - 1) == 0:
            logits = layer(h, 784, 'logits', bn=False)
            return z, z_post, logits
        else:
            x_m = layer(h, x_size, 'mean')
            x_v = layer(h, x_size, 'variance', activation=tf.nn.softplus)
            x_prior = (x_m, x_v)
            return z, z_post, x_prior

# Ladder VAE set-up
tf.reset_default_graph()
phase = Placeholder(None, tf.bool, name='phase')
x = Placeholder((None, 784), name='x')
with tf.name_scope('z0'):
    z0 = tf.cast(tf.greater(x, tf.random_uniform(tf.shape(x), 0, 1)), tf.float32)

# encode
z1_like = encode_block(z0, 512, 64, idx=1)
z2_like = encode_block(z1_like[0], 256, 32, idx=2)
z3_like = encode_block(z2_like[0], 128, 16, idx=3)
# decode
z3_prior = (Constant(0), Constant(1))
z3, z3_post, z2_prior = decode_block(z3_like, None, 128, 32, idx=3)
z2, z2_post, z1_prior = decode_block(z2_like, z2_prior, 256, 64, idx=2)
z1, z1_post, z0_logits = decode_block(z1_like, z1_prior, 512, 784, idx=1)

with tf.name_scope('loss'):
    with tf.name_scope('recon'):
        recon = -log_bern(z0, z0_logits)
    with tf.name_scope('kl1'):
        kl1   = -log_norm(z1, *z1_prior) + log_norm(z1, *z1_post)
    with tf.name_scope('kl2'):
        kl2   = -log_norm(z2, *z2_prior) + log_norm(z2, *z2_post)
    with tf.name_scope('kl3'):
        kl3   = -log_norm(z3, *z3_prior) + log_norm(z3, *z3_post)
    loss  = tf.reduce_mean(recon + kl1 + kl2 + kl3)

lr = Placeholder(None, name='lr')
update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
with tf.control_dependencies(update_ops):
    # Ensures that we execute the update_ops before performing the train_step
    train_step = tf.train.AdamOptimizer(lr).minimize(loss)

sess = tf.Session()
tf.scalar_summary('learning_rate', lr)
tf.scalar_summary('loss', loss)
merged = tf.merge_all_summaries()
train_writer = tf.train.SummaryWriter('results/lvae{:d}/train'.format(args.run), sess.graph)
test_writer = tf.train.SummaryWriter('results/lvae{:d}/test'.format(args.run))
sess.run(tf.initialize_all_variables())

iterep = 50000/args.bs
for i in range(iterep * 2000):
    x_train, y_train = mnist.train.next_batch(args.bs)
    sess.run(train_step,
             feed_dict={'x:0': x_train,
                        'phase:0': True,
                        'lr:0': args.lr})
    progbar(i, iterep)
    if (i + 1) %  iterep == 0:
        epoch = (i + 1)/iterep
        summary0, loss0 = sess.run([merged, loss], feed_dict={'x:0': mnist.train.images,
                                                              'phase:0': False,
                                                              'lr:0': args.lr})
        train_writer.add_summary(summary0, epoch)
        summary1, loss1 = sess.run([merged, loss], feed_dict={'x:0': mnist.test.images,
                                                              'phase:0': False,
                                                              'lr:0': args.lr})
        test_writer.add_summary(summary1, epoch)
        print ("Epoch={:d}. Learning rate={:2f}. Training loss={:.2f}. Test loss={:.2f}"
               .format(epoch, args.lr, loss0, loss1))

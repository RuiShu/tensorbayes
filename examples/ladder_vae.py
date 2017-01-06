"""Implementation of the Ladder Variational Autoencoder
The default settings should be able to achieve a lower bound of ~85, which is
consistent with what their implementation. Please bear in mind the set-up is a
little different though, so direct comparison should be taken with a grain of
salt.
"""
import argparse
parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("-run", type=int, required=True, help="Run index. Use 0 if first run.")
parser.add_argument("-bs", type=int, help="Minibatch size.", default=256)
parser.add_argument("-lr", type=float, help="Learning rate.", default=5e-4)
parser.add_argument("-nonlin", type=str, help="Activation function.", default='elu')
parser.add_argument("-eps", type=float, help="Distribution epsilon.", default=1e-5)
parser.add_argument("-save_dir", type=str, help="Save model directory.", default='/scratch/users/rshu15')
parser.add_argument("-n_checks", type=int, help="Number of check points.", default=100)
args = parser.parse_args()

import sys
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='3'
import numpy as np
import tensorflow as tf
from tensorflow.contrib.framework import arg_scope
from tensorflow.examples.tutorials.mnist import input_data
import tensorbayes as tb
from tensorbayes.layers import *
from tensorbayes.distributions import log_bernoulli_with_logits, log_normal

if args.nonlin == 'relu':
    activate = tf.nn.relu
elif args.nonlin == 'elu':
    activate = tf.nn.elu
else:
    raise Exception("Unexpected nonlinearity arg")
args.save_dir = args.save_dir.rstrip('/')
log_file = 'results/lvae{:d}.csv'.format(args.run)
model_dir = '{:s}/{:s}'.format(args.save_dir, log_file.rstrip('.csv'))
log_bern = lambda x, logits: log_bernoulli_with_logits(x, logits, args.eps)
log_norm = lambda x, mu, var: log_normal(x, mu, var, 0.0)
writer = tb.FileWriter(log_file, args=args, pipe_to_sys=True)

# Convenience layers and graph blocks
def name(index, suffix):
    return 'z{:d}'.format(index) + '_' + suffix

def encode_block(x, h_size, z_size, idx):
    with tf.variable_scope(name(idx, 'encode')):
        h = dense(x, h_size, 'layer1', activation=activate)
        h = dense(h, h_size, 'layer2', activation=activate)
    with tf.variable_scope(name(idx, 'encode/likelihood')):
        z_m = dense(h, z_size, 'mean')
        z_v = dense(h, z_size, 'var', activation=tf.nn.softplus) + args.eps
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
        h = dense(z, h_size, 'layer1', activation=activate)
        h = dense(h, h_size, 'layer2', activation=activate)
    with tf.variable_scope(name(idx - 1, 'decode/prior')):
        if (idx - 1) == 0:
            logits = dense(h, 784, 'logits', bn=False)
            return z, z_post, logits
        else:
            x_m = dense(h, x_size, 'mean')
            x_v = dense(h, x_size, 'var', activation=tf.nn.softplus) + args.eps
            x_prior = (x_m, x_v)
            return z, z_post, x_prior

# Ladder VAE set-up
tf.reset_default_graph()
phase = Placeholder(None, tf.bool, name='phase')
lr = Placeholder(None, name='lr')
x = Placeholder((None, 784), name='x')
with tf.name_scope('z0'):
    z0 = tf.cast(tf.greater(x, tf.random_uniform(tf.shape(x), 0, 1)),
                 tf.float32)

# encode
with arg_scope([dense], bn=True, phase=phase):
    z1_like = encode_block(z0, 512, 64, idx=1)
    z2_like = encode_block(z1_like[0], 256, 32, idx=2)
    z3_like = encode_block(z2_like[0], 128, 16, idx=3)
    z4_like = encode_block(z3_like[0],  64,  8, idx=4)
    # decode
    z4_prior = (Constant(0), Constant(1))
    z4, z4_post, z3_prior = decode_block(z4_like, None,  64, 16, idx=4)
    z3, z3_post, z2_prior = decode_block(z3_like, z3_prior, 128, 32, idx=3)
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
    with tf.name_scope('kl4'):
        kl4   = -log_norm(z4, *z4_prior) + log_norm(z4, *z4_post)
    per_sample_loss  = recon + kl1 + kl2 + kl3 + kl4
    loss = tf.reduce_mean(per_sample_loss)

update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
with tf.name_scope('gradients'):
    optimizer = tf.train.AdamOptimizer(lr)
    clipped, grad_norm = tb.tbutils.clip_gradients(optimizer, loss,
                                                   max_clip=0.9,
                                                   max_norm=4)
    with tf.control_dependencies(update_ops):
        # Ensures that we execute the update_ops before performing the
        # train_step
        train_step = optimizer.apply_gradients(clipped)

# initialize and save model
sess = tf.Session()
sess.run(tf.global_variables_initializer())
saver = tf.train.Saver()
if not os.path.exists(model_dir):
    os.makedirs(model_dir)

mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
writer.add_var('train_loss', '{:10.3f}', loss)
writer.add_var('test_loss', '{:10.3f}')
writer.add_var('epoch', '{:>10d}')
writer.initialize()
iterep = len(mnist.train.images)/args.bs
for i in range(iterep * 2000):
    x_train, y_train = mnist.train.next_batch(args.bs)
    _, g, l, p = sess.run([train_step, grad_norm, loss, per_sample_loss],
                          feed_dict={'x:0': x_train,
                                     'phase:0': True,
                                     'lr:0': args.lr})
    message = "grad_norm: {:.2e}. loss: {:.2e}".format(g, l)
    end_epoch, epoch = tb.utils.progbar(i, iterep, message, bar_length=5)
    if np.isnan(g) or np.isnan(l):
        print "NaN detected. Printing per-sample-loss."
        print p
        quit()
    if end_epoch:
        tr_values = sess.run(writer.tensors,
                             feed_dict={'x:0': mnist.train.images,
                                        'phase:0': False,
                                        'lr:0': args.lr})
        te_values = sess.run(writer.tensors,
                             feed_dict={'x:0': mnist.test.images,
                                        'phase:0': False,
                                        'lr:0': args.lr})
        writer.write(tensor_values=tr_values,
                     values = te_values + [epoch])
        if epoch % args.n_checks == 0:
            path = saver.save(sess, '{:s}/model.ckpt'
                              .format(model_dir, args.run))
            print "Saved model to {:s}".format(path)

path = saver.save(sess, '{:s}/model.ckpt'.format(model_dir, args.run))
print "Saved model to {:s}".format(path)

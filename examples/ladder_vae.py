"""Implementation of the Ladder Variational Autoencoder

The default settings should be able to achieve a lower bound of ~85, which is
consistent with what their implementation. Please bear in mind the set-up is a
little different though, so direct comparison should be taken with a grain of
salt.
"""
import os
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

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("-run", type=int, help="Run index. Use 0 if first run.")
parser.add_argument("-bs", type=int, help="Minibatch size.", default=256)
parser.add_argument("-lr", type=float, help="Learning rate.", default=5e-4)
parser.add_argument("-nonlin", type=str, help="Activation function.", default='elu')
parser.add_argument("-eps", type=float, help="Distribution epsilon.", default=1e-8)
parser.add_argument("-save_dir", type=str, help="Save model directory.", default='/scratch/users/rshu15')
parser.add_argument("-n_checks", type=int, help="Number of check points.", default=100)
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
    raise Exception("Unexpected nonlinearity arg")
args.save_dir = args.save_dir.rstrip('/')
model_dir = '{:s}/results/lvae{:d}'.format(args.save_dir, args.run)
log_bern = lambda x, logits: log_bernoulli_with_logits(x, logits, args.eps)
log_norm = lambda x, mu, var: log_normal(x, mu, var, 0.0)
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

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
        z_v = layer(h, z_size, 'variance', activation=tf.nn.softplus) + args.eps
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
            x_v = layer(h, x_size, 'variance', activation=tf.nn.softplus) + args.eps
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
    per_sample_loss  = recon + kl1 + kl2 + kl3
    loss = tf.reduce_mean(per_sample_loss)

lr = Placeholder(None, name='lr')
update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)

# optimization and gradient clipping
optimizer = tf.train.AdamOptimizer(lr)

with tf.name_scope('gradients'):
    grads_and_vars = optimizer.compute_gradients(loss)
    grads = [g for g, _ in grads_and_vars]
    max_clip = 0.9
    max_norm = 4
    grads, global_grad_norm = tf.clip_by_global_norm(grads, max_norm)
    clipped_grads_and_vars = []
    for i in xrange(len(grads_and_vars)):
        g = tf.clip_by_value(grads[i], -max_clip, max_clip)
        v = grads_and_vars[i][1]
        clipped_grads_and_vars += [(g, v)]

    with tf.control_dependencies(update_ops):
        # Ensures that we execute the update_ops before performing the train_step
        train_step = optimizer.apply_gradients(clipped_grads_and_vars)

sess = tf.Session()
tf.scalar_summary('learning_rate', lr)
tf.scalar_summary('loss', loss)
merged = tf.merge_all_summaries()
train_writer = tf.train.SummaryWriter('results/lvae{:d}/train'.format(args.run), sess.graph)
test_writer = tf.train.SummaryWriter('results/lvae{:d}/test'.format(args.run))
sess.run(tf.initialize_all_variables())

# save model
saver = tf.train.Saver()
if not os.path.exists(model_dir):
    os.mkdir(model_dir)
iterep = 50000/args.bs
for i in range(iterep * 2000):
    x_train, y_train = mnist.train.next_batch(args.bs)
    _, gn, l, psl = sess.run([train_step, global_grad_norm, loss, per_sample_loss],
                        feed_dict={'x:0': x_train,
                                   'phase:0': True,
                                   'lr:0': args.lr})
    prog_bar_message = "grad_norm: {:.2e}. loss: {:.2e}".format(gn, l)
    progbar(i, iterep, prog_bar_message, bar_length=5)
    if np.isnan(gn) or np.isnan(l):
        print "NaN detected. Printing per-sample-loss."
        print psl
        quit()
    if (i + 1) % (iterep * args.n_checks) == 0:
        save_path = saver.save(sess, '{:s}/model.ckpt'.format(model_dir, args.run))
        print "Saved model to {:s}".format(save_path)
    if (i + 1) %  iterep == 0:
        epoch = (i + 1)/iterep
        summary0, loss0 = sess.run([merged, loss], feed_dict={'x:0': mnist.train.images,
                                                              'phase:0': False,
                                                              'lr:0': args.lr})
        train_writer.add_summary(summary0, epoch)
        train_writer.flush()
        summary1, loss1 = sess.run([merged, loss], feed_dict={'x:0': mnist.test.images,
                                                              'phase:0': False,
                                                              'lr:0': args.lr})
        test_writer.add_summary(summary1, epoch)
        test_writer.flush()
        print ("Epoch={:d}. Learning rate={:2f}. Training loss={:.2f}. Test loss={:.2f}"
               .format(epoch, args.lr, loss0, loss1))

save_path = saver.save(sess, '{:s}/results/lvae{:d}/model.ckpt'.format(args.save_dir, args.run))
print "Saved final model to {:s}".format(save_path)

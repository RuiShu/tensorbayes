import os
import sys
import time

def progbar(i, iter_per_epoch, message='', bar_length=50):
    j = (i % iter_per_epoch) + 1
    perc = int(100. * j / iter_per_epoch)
    prog = ''.join(['='] * (bar_length * perc / 100))
    template = "\r[{:" + str(bar_length) + "s}] {:3d}%. {:s}"
    string = template.format(prog, perc, message)
    sys.stdout.write(string)
    sys.stdout.flush()
    end_epoch = j == iter_per_epoch
    if end_epoch:
        sys.stdout.write('\r{:100s}\r'.format(''))
        sys.stdout.flush()
    return end_epoch, (i + 1)/iter_per_epoch

class SummaryWriter(object):
    def __init__(self, logdir, overwrite=False):
        self.written = False
        self.logdir = os.path.dirname(logdir + '/')
        if os.path.exists(self.logdir):
            if not overwrite:
                raise Exception("Overwriting existing log directory is not allowed "
                                "unless overwrite=True")
        else:
            os.makedirs(self.logdir)
        self.tensor_names = []
        self.tensors = []
        self.tensor_formats = []
        self.suffix_header = ''
        train_name = os.path.join(self.logdir, 'train_summary.csv')
        test_name = os.path.join(self.logdir, 'test_summary.csv')
        self.f_train = open(train_name, 'w', 0)
        self.f_test = open(test_name, 'w', 0)
        self.f = None

    def add_summary(self, name, tensor, tensor_format):
        self.tensor_names += [name]
        self.tensors += [tensor]
        self.tensor_formats += [tensor_format]

    def add_suffix_header(self, suffix_header):
        self.suffix_header = suffix_header

    def write(self, string):
        self.f.write(string + '\n')
        self.f.flush()

    def write_summary(self, f_type, values, suffix=''):
        if self.written is False:
            self.f = self.f_train
            self.write(','.join(self.tensor_names) + self.suffix_header)
            self.f = self.f_test
            self.write(','.join(self.tensor_names) + self.suffix_header)
        if f_type not in {'train', 'test'}:
            raise Exception('Unable to identify f_type')
        self.f = self.f_train if f_type == 'train' else self.f_test
        self.write(','.join(self.tensor_formats).format(*values) + suffix)
        self.written = True
        self.f = None

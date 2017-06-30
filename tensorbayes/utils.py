import os
import sys
import time

def progbar(i, iter_per_epoch, message='', bar_length=50, display=True):
    j = (i % iter_per_epoch) + 1
    end_epoch = j == iter_per_epoch
    if display:
        perc = int(100. * j / iter_per_epoch)
        prog = ''.join(['='] * (bar_length * perc / 100))
        template = "\r[{:" + str(bar_length) + "s}] {:3d}%. {:s}"
        string = template.format(prog, perc, message)
        sys.stdout.write(string)
        sys.stdout.flush()
        if end_epoch:
            sys.stdout.write('\r{:100s}\r'.format(''))
            sys.stdout.flush()
    return end_epoch, (i + 1)/iter_per_epoch

class FileWriter(object):
    def __init__(self, log_file, args=None,
                 overwrite=False, pipe_to_sys=True):
        self.written = False
        self.log_file = log_file
        self.pipe = pipe_to_sys
        self.args = args
        # non-tensorflow values to be stored
        self.names = []
        self.formats = []
        # tf tensor values
        self.tensor_names = []
        self.tensors = []
        self.tensor_formats = []
        # check file existence, then create file
        if os.path.exists(self.log_file) and not overwrite:
            raise Exception("Overwriting existing log directory is "
                            "not allowed unless overwrite=True")
        log_dir = os.path.dirname(self.log_file)
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
        self.f = open(self.log_file, 'w', 0)
        if args is not None:
            # create file
            # write args
            v_dict = vars(self.args)
            string = '# ArgParse Values:'
            self._write(string)
            for k in v_dict:
                string = '# {:s}: {:s}'.format(str(k), str(v_dict[k]))
                self._write(string)

    @staticmethod
    def list_args(args):
        v_dict = vars(args)
        print '# ArgParse Values:'
        for k in v_dict:
            print '# {:s}: {:s}'.format(str(k), str(v_dict[k]))

    def initialize(self):
        # create header name
        self.header = ','.join(self.tensor_names + self.names)
        self._write(self.header)

    def add_var(self, name, var_format, tensor=None):
        if tensor is None:
            self.names += [name]
            self.formats += [var_format]
        else:
            self.tensor_names += [name]
            self.tensors += [tensor]
            self.tensor_formats += [var_format]

    def write(self, tensor_values=[], values=[]):
        values = tensor_values + values
        string = ','.join(self.tensor_formats + self.formats).format(*values)
        self._write(string, is_summary=True, pipe=True)

    def _write(self, string, is_summary=False, pipe=False):
        self.f.write(string + '\n')
        self.f.flush()
        if self.pipe and pipe:
            if is_summary:
                print(self.header)
            print(string)

class TensorDict(object):
    def __init__(self, d={}):
        self.__dict__ = dict(d)

    def __iter__(self):
        return iter(self.__dict__)

    def __contains__(self, key):
        return key in self.__dict__

    def __getitem__(self, key):
        return self.__dict__[key]

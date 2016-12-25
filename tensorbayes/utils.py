import sys

def progbar(i, iter_per_epoch):
    j = (i % iter_per_epoch) + 1
    perc = int(100. * j / iter_per_epoch)
    prog = ''.join(['='] * (perc/2))
    string = "\r[{:50s}] {:3d}%".format(prog, perc)
    sys.stdout.write(string)
    sys.stdout.flush()
    if j == iter_per_epoch:
        sys.stdout.write('\r{:100s}\r'.format(''))
        sys.stdout.flush()

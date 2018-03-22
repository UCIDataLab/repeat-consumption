"""
This file contains methods that wrap around numpy and pickle for easy IO.
Save methods, create the directory path if needed and change the permissions to be shared by group.
"""
import os
import sys
import time

import cPickle as pickle
import numpy as np


def make_go_rw(filename, change_perm):
    if change_perm:
        os.chmod(filename, 0770)


def make_dir(filename):
    dir_path = os.path.dirname(filename)
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)


def save_pickle(filename, obj, verbose=False, other_permission=True):
    make_dir(filename)
    if verbose:
        print '--> Saving ', filename, ' with pickle was ',
        sys.stdout.flush()
    t = time.time()
    with open(filename, 'wb') as gfp:
        pickle.dump(obj, gfp, protocol=pickle.HIGHEST_PROTOCOL)
        gfp.close()

    if verbose:
        print '%.3f s' % (time.time() - t)
    make_go_rw(filename, other_permission)


def load_pickle(filename, verbose=False):
    if verbose:
        print '--> Loading ', filename, ' with pickle was ',
        sys.stdout.flush()
    t = time.time()
    with open(filename, 'rb') as gfp:
        r = pickle.load(gfp)

    if verbose:
        print '%.3f s' % (time.time() - t)
    return r


def load_txt(filename, delimiter=',', verbose=True):
    if verbose:
        print '--> Loading ', filename, ' with np.loadtxt was ',
    sys.stdout.flush()
    t = time.time()
    d = np.loadtxt(filename, delimiter=delimiter)
    if verbose:
        print '%.3f s' % (time.time() - t)
    return d

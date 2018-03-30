import numpy as np


def tanh_deriv(x):
        return 1/np.cosh(x)**2

def sigmoid(x):
    return 1/(1 + np.exp(-x))

def sigmoid_deriv(x):
    ex = np.exp(-x)
    return ex/(ex + 1)**2

def shuffle_sync(*args):
    perm = np.random.permutation(args[0].shape[0])
    return [vec[perm] for vec in args]

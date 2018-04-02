import numpy as np


def tanh_deriv(x):
        return 1/np.cosh(x)**2

def sigmoid(x):
    return 1/(1 + np.exp(-x))

def sigmoid_deriv_raw(x):
    ex = np.exp(-x)
    return ex/(ex + 1)**2

def sigmoid_deriv(sx):
    return sx*(1 - sx)

def softmax(x):
    t = x - np.max(x, axis=-1, keepdims=True)
    et = np.exp(t)
    return et/np.sum(et, axis=-1, keepdims=True)

def shuffle_sync(seqs):
    perm = np.random.permutation(seqs[0].shape[0])
    return [s[perm] for s in seqs]

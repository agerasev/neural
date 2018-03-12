import numpy as np


def tanh_deriv(x):
        return 1/np.cosh(x)**2

def sigmoid(x):
    return 1/(1 + np.exp(-x))

def sigmoid_deriv(x):
    ex = np.exp(-x)
    return ex/(ex + 1)**2

def rms_loss(y, r):
    d = (y - r)**2
    return np.sum(d, axis=None)/(2*r.shape[0])

def rms_deriv(y, r):
    return y - r

def softmax(y):
    y = y - np.max(y)
    ey = np.exp(y)
    return ey/np.sum(ey, axis=-1, keepdims=True)

def ce_loss(y, ir):
    mask = np.equal(ir.reshape(-1, 1), np.arange(y.shape[-1]))
    return np.sum(-np.log(y[mask]))/ir.shape[0]

def ce_softmax_deriv(y, ir):
    mask = np.equal(ir.reshape(-1, 1), np.arange(y.shape[-1]))
    dy = np.copy(y)
    dy[mask] -= 1.0
    return dy

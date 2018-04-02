import numpy as np

from .util import *


class Loss:
    def __init__(self):
        pass

    def forward(self, x, r=None):
        raise NotImplementedError()

    def backward(self, m):
        raise NotImplementedError()

class RMS(Loss):
    def __init__(self):
        super().__init__()

    def forward(self, x, r):
        loss = np.sum((x - r)**2, axis=None)/(2*r.shape[0])
        cache = (x, r)

        return x, loss, cache

    def backward(self, cache):
        x, r = cache
        dy = x - r
        return dy

class Softmax(Loss):
    def __init__(self):
        super().__init__()

    def forward(self, x, r):
        y = softmax(x)

        mask = np.equal(r.reshape(-1, 1), np.arange(y.shape[-1]))
        loss = np.sum(-np.log(y[mask]))/r.shape[0]
        cache = (y, mask)

        return x, loss, cache

    def backward(self, cache):
        y, mask = cache
        dy = np.copy(y)
        dy[mask] -= 1.0
        return dy

class Binary(Loss):
    def __init__(self):
        super().__init__()

    def forward(self, x, r):
        mask = np.equal(r, np.arange(x.shape[-1]))
        y = sigmoid(-x)
        y[mask] = 1.0 - y[mask]
        loss = np.sum(-np.log(y), axis=None)/r.shape[0]
        cache = (x, mask)
        return x, loss, cache

    def backward(self, cache):
        x, mask = cache
        dy = sigmoid(x)
        dy[mask] -= 1.0
        return dy

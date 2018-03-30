import numpy as np


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

    def forward(self, x, r=None):
        y = x
        if r is not None:
            loss = np.sum((y - r)**2, axis=None)/(2*r.shape[0])
            cache = (y, r)
        else:
            loss = None
            cache = None
        return y, loss, cache

    def backward(self, cache):
        y, r = cache
        dy = y - r
        return dy

class Softmax(Loss):
    def __init__(self):
        super().__init__()

    def forward(self, x, r=None):
        t = x - np.max(x)
        et = np.exp(t)
        y = et/np.sum(et, axis=-1, keepdims=True)
        if r is not None:
            mask = np.equal(r.reshape(-1, 1), np.arange(y.shape[-1]))
            loss = np.sum(-np.log(y[mask]))/r.shape[0]
            cache = (y, r, mask)
        else:
            loss = None
            cache = None
        return x, loss, cache

    def backward(self, cache):
        y, r, mask = cache
        dy = np.copy(y)
        dy[mask] -= 1.0
        return dy

import numpy as np

from .layer import Affine


class SparseAffine(Affine):
    def __init__(self, sx, sy, mag=None):
        super().__init__(sx, sy, mag)
        self.ios = (2, 1)

    def forward(self, xs):
        xv, xi = xs
        #print(xv.shape, xi.shape, self.W[xi.reshape(-1)].shape)
        y = xv*self.W[xi.reshape(-1)]
        #print(y.shape)
        return y, xs

    def backward(self, grad, cache, dy):
        xv, xi = cache
        #print(xv.shape, xi.shape, dy.shape, self.W[xi.reshape(-1)].shape)
        grad.W[xi.reshape(-1)] += xv*dy#/xi.shape[0]
        dxv = np.sum(self.W[xi.reshape(-1)]*dy, axis=-1)
        #print(dxv.shape)
        return dxv, xi

class AffineSparse(Affine):
    def __init__(self, sx, sy, mag=None):
        super().__init__(sy, sx, mag)
        self.ios = (2, 1)

    def forward(self, xyi):
        x, yi = xyi
        #print(x.shape, yi.shape, self.W[yi].shape)
        yv = np.sum(x.reshape(x.shape[0], 1, x.shape[1])*self.W[yi], axis=-1)
        #print(yv.shape)
        return yv, xyi

    def backward(self, grad, cache, dyv):
        x, yi = cache
        #print(x.shape, yi.shape, dyv.shape, grad.W[yi].shape)
        grad.W[yi] += x.reshape(x.shape[0], 1, x.shape[1])*dyv.reshape(*dyv.shape, 1)#/yi.shape[0]
        dx = np.sum(self.W[yi]*dyv.reshape(*dyv.shape, 1), axis=1)
        #print(dx.shape)
        return dx, yi

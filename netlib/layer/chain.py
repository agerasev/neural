import numpy as np

from .layer import *


class ChainParam(Param):
    def __init__(self, params):
        super().__init__()
        self.params = params

    def __iter__(self):
        for p in self.params:
            yield from p

    def newgrad(self):
        return ChainParam([p.newgrad() for p in self.params])

class Chain(Layer, Param):
    def __init__(self, layers, **kwagrs):
        Layer.__init__(self, **kwagrs)
        Param.__init__(self)
        self.layers = layers

    def forward(self, x):
        cache = []
        for layer in self.layers:
            x, c = layer.forward(x)
            cache.append(c)
        return x, c

    def backward(self, grad, cache, dy):
        for l, g, c in reversed(list(zip(self.layers, grad, cache))):
            dy = l.backward(g, c, dy)
        return dy

    def __iter__(self):
        for l in self.layers:
            yield from l

    def newgrad(self):
        return ChainParam([l.newgrad() for l in self.layers])


class AffineBias(Chain):
    def __init__(self, sx, sy, **kwagrs):
        layers = [
            Affine(sx, sy, **kwagrs),
            Bias(sy, **kwagrs)
        ]
        super().__init__(layers, **kwagrs)

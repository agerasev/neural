import numpy as np

from .util import *


class Param:
    def __init__(self):
        pass

    def __iter__(self):
        raise NotImplementedError()

class Layer(Param):
    def __init__(self):
        Param.__init__(self)
        self.ios = (1, 1)
    
    def forward(self, x):
        raise NotImplementedError()

    def backward(self, grad, cache, dy):
        raise NotImplementedError()

    def newgrad(self):
        raise NotImplementedError()

class AffineParam(Param):
    def __init__(self, W):
        super().__init__()
        self.W = W

    def __iter__(self):
        yield self.W

class Affine(Layer, AffineParam):
    def __init__(self, sx, sy, mag=1):
        Layer.__init__(self)
        AffineParam.__init__(self, (mag/(sx + sy))*np.random.randn(sx, sy))

    def forward(self, x):
        return np.tensordot(x, self.W, axes=(-1, 0)), x

    def backward(self, grad, cache, dy):
        grad.W += np.tensordot(cache, dy, axes=(0, 0))/dy.shape[0]
        return np.tensordot(dy, self.W, axes=(-1, 1))

    def newgrad(self):
        return AffineParam(np.zeros_like(self.W))

class BiasParam(Param):
    def __init__(self, b):
        Param.__init__(self)
        self.b = b

    def __iter__(self):
        yield self.b

class Bias(Layer, BiasParam):
    def __init__(self, s):
        Layer.__init__(self)
        BiasParam.__init__(self, np.zeros(s, dtype=np.float64))

    def forward(self, x):
        return self.b + x, None

    def newgrad(self):
        return BiasParam(np.zeros_like(self.b))

    def backward(self, grad, cache, dy):
        grad.b += np.sum(dy, axis=0)/dy.shape[0]
        return dy

class EmptyParam(Param):
    def __init__(self):
        Param.__init__(self)

    def __iter__(self):
        return
        yield

class EmptyLayer(Layer, EmptyParam):
    def __init__(self):
        Layer.__init__(self)
        EmptyParam.__init__(self)

    def newgrad(self):
        return EmptyParam()

class ReLU(EmptyLayer):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return np.maximum(x, 0.0), x

    def backward(self, grad, cache, dy):
        return dy*np.greater(cache, 0.0)

class Tanh(EmptyLayer):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return np.tanh(x), x

    def backward(self, grad, cache, dy):
        return dy*tanh_deriv(cache)

class Sigmoid(EmptyLayer):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return sigmoid(x), x

    def backward(self, grad, cache, dy):
        return dy*sigmoid_deriv(cache)

class Uniform(EmptyLayer):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x, None

    def backward(self, grad, cache, dy):
        return dy

class Product(EmptyLayer):
    def __init__(self):
        super().__init__()
        self.ios = (2, 1)

    def forward(self, x):
        return x[0]*x[1], x

    def backward(self, grad, cache, dy):
        return cache[1]*dy, cache[0]*dy

import numpy as np

from ..util import *


class Param:
    def __init__(self):
        pass

    def __iter__(self):
        raise NotImplementedError()

    def newgrad(self):
        raise NotImplementedError()        

class Layer():
    def __init__(self, **kwargs):
        self.ios = (1, 1)
        self.dtype = kwargs.get("dtype", np.float64)

    def forward(self, x):
        raise NotImplementedError()

    def backward(self, grad, cache, dy):
        raise NotImplementedError()

class AffineParam(Param):
    def __init__(self, W):
        super().__init__()
        self.W = W

    def __iter__(self):
        yield self.W

    def newgrad(self):
        AffineParam(np.zeros_like(self.W))

class Affine(Layer, AffineParam):
    def __init__(self, sx, sy, **kwargs):
        Layer.__init__(self, **kwargs)
        data = np.random.randn(sx, sy).astype(self.dtype)
        data *= kwargs.get("mag", 1.0/(sx + sy))
        AffineParam.__init__(self, data)

    def forward(self, x):
        return np.tensordot(x, self.W, axes=(-1, 0)), x

    def backward(self, grad, cache, dy):
        grad.W += np.tensordot(cache, dy, axes=(0, 0))#/dy.shape[0]
        return np.tensordot(dy, self.W, axes=(-1, 1))

class BiasParam(Param):
    def __init__(self, b):
        Param.__init__(self)
        self.b = b

    def __iter__(self):
        yield self.b

    def newgrad(self):
        return BiasParam(np.zeros_like(self.b))

class Bias(Layer, BiasParam):
    def __init__(self, s, **kwargs):
        Layer.__init__(self, **kwargs)
        BiasParam.__init__(self, np.zeros(s, dtype=self.dtype))

    def forward(self, x):
        return self.b + x, None

    def backward(self, grad, cache, dy):
        grad.b += np.sum(dy, axis=0)#/dy.shape[0]
        return dy

class NoParam(Param):
    def __init__(self):
        Param.__init__(self)

    def __iter__(self):
        return
        yield

    def newgrad(self):
        return NoParam()

class ReLU(Layer, NoParam):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def forward(self, x):
        return np.maximum(x, 0.0), x

    def backward(self, grad, cache, dy):
        return dy*np.greater(cache, 0.0)

class Tanh(Layer, NoParam):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def forward(self, x):
        return np.tanh(x), x

    def backward(self, grad, cache, dy):
        return dy*tanh_deriv(cache)

class Sigmoid(Layer, NoParam):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def forward(self, x):
        sx = sigmoid(x)
        return sx, sx

    def backward(self, grad, cache, dy):
        return dy*sigmoid_deriv(cache)

class Uniform(Layer, NoParam):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def forward(self, x):
        return x, None

    def backward(self, grad, cache, dy):
        return dy

class Product(Layer, NoParam):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.ios = (2, 1)

    def forward(self, x):
        return x[0]*x[1], x

    def backward(self, grad, cache, dy):
        return cache[1]*dy, cache[0]*dy

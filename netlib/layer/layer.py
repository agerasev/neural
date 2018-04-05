import numpy as np

from ..util import *


class _Param:
    def __init__(self):
        pass

    def __iter__(self):
        raise NotImplementedError()

    def newgrad(self):
        raise NotImplementedError()        

class _Layer:
    def __init__(self):
        pass
    
    def forward(self, x):
        raise NotImplementedError()

    def backward(self, grad, cache, dy):
        raise NotImplementedError()

class Layer():
    def __init__(self, **kwargs):
        self.ios = (1, 1)
        self.dtype = kwargs.get("dtype", np.float64)

class _AffineParam(_Param):
    def __init__(self, W):
        super().__init__()
        self.W = W

    def __iter__(self):
        yield self.W

    def newgrad(self):
        _AffineParam(np.zeros_like(self.W))

class _Affine(_Layer):
    def __init__(self):
        super().__init__()

    def _initparam(sx, sy, mag, dtype):
        if mag is None:
            mag = 1.0/(sx + sy)
        randn = np.random.randn(sx, sy, dtype=dtype)
        return mag*randn

    def forward(self, x):
        return np.tensordot(x, self.W, axes=(-1, 0)), x

    def backward(self, grad, cache, dy):
        grad.W += np.tensordot(cache, dy, axes=(0, 0))#/dy.shape[0]
        return np.tensordot(dy, self.W, axes=(-1, 1))

class Affine(Layer, _Affine, _AffineParam):
    def __init__(self, sx, sy, **kwargs):
        Layer.__init__(self, **kwargs)
        _Affine.__init__(self)
        _AffineParam.__init__(
            self,
            _Affine._ip(
                sx, sy, 
                kwargs.get("mag", None), 
                dtype=self.dtype
            )
        )

class _BiasParam(_Param):
    def __init__(self, b):
        _Param.__init__(self)
        self.b = b

    def __iter__(self):
        yield self.b

    def newgrad(self):
        return _BiasParam(np.zeros_like(self.b))

class _Bias(_Layer):
    def __init__(self):
        super().__init__()

    def _ip(self, s, dtype):
        return np.zeros(s, dtype=dtype)

    def forward(self, x):
        return self.b + x, None

    def backward(self, grad, cache, dy):
        grad.b += np.sum(dy, axis=0)#/dy.shape[0]
        return dy

class Bias(Layer, _Bias, _BiasParam):
    def __init__(self, s, **kwargs):
        Layer.__init__(self, **kwargs)
        _Bias.__init__(self)
        BiasParam.__init__(self, _Bias._ip(s, self.dtype))

class _AffineBiasParam(_AffineParam, _BiasParam):
    def __init__(self, W, b):
        _AffineParam.__init__(self, W)
        _BiasParam.__init__(self, b)

    def __iter__(self):
        yield self.W
        yield self.b

    def newgrad(self):
        return _AffineBiasParam(
            *[np.zeros_like(w) for w in [self.W, self.b]]
        )

class _AffineBias(_Affine, _Bias):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        t, m0 = _Affine.forward(self, x)
        y, m1 = _Bias.forward(self, t)
        return y, (m0, m1)

    def backward(self, grad, cache, dy):
        m0, m1 = cache
        dt = _Bias.backward(self, grad, m1, dy)
        dx = _Affine.backward(self, grad, m0, dt)
        return dx

class _AffineBias(_Layer, AffineBiasParam):
    def __init__(self, sx, sy, mag=None):
        Affine.__init__(self, sx, sy, mag=None)
        Bias.__init__(self, sy)
        _Param.__init__(self)

    

class EmptyParam(_Param):
    def __init__(self):
        _Param.__init__(self)

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
        sx = sigmoid(x)
        return sx, sx

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

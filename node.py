import numpy as np
from util import *

class Param:
    def __init__(self):
        pass

    def __iter__(self):
        raise NotImplementedError()

class Node(Param):
    def __init__(self):
        pass
    
    def feed(self, x):
        raise NotImplementedError()
    
    def feed_mem(self, x):
        raise NotImplementedError()
    
    def newgrad(self):
        raise NotImplementedError()

    def backprop(self, grad, m, dy):
        raise NotImplementedError()

class AffineParam(Param):
    def __init__(self, W):
        Param.__init__(self)
        self.W = W

    def __iter__(self):
        yield self.W

class Affine(Node, AffineParam):
    def __init__(self, sx, sy, mag=1e-2):
        Node.__init__(self)
        AffineParam.__init__(self, mag*np.random.randn(sx, sy))

    def feed(self, x):
        return np.tensordot(x, self.W, axes=(-1, 0))

    def feed_mem(self, x):
        return self.feed(x), x

    def newgrad(self):
        return AffineParam(np.zeros_like(self.W))

    def backprop(self, grad, m, dy):
        grad.W += np.tensordot(m, dy, axes=(0, 0))/dy.shape[0]
        return np.tensordot(dy, self.W, axes=(-1, 1))

    def __iter__(self):
        yield self.W

class BiasParam(Param):
    def __init__(self, b):
        Param.__init__(self)
        self.b = b

    def __iter__(self):
        yield self.b

class Bias(Node, BiasParam):
    def __init__(self, s, mag=1e-2):
        Node.__init__(self)
        BiasParam.__init__(self, mag*np.random.randn(s))

    def feed(self, x):
        return self.b + x

    def feed_mem(self, x):
        return self.feed(x), None

    def newgrad(self):
        return BiasParam(np.zeros_like(self.b))

    def backprop(self, grad, m, dy):
        grad.b += np.sum(dy, axis=0)
        return dy

class _EmptyParam(Param):
    def __init__(self):
        Param.__init__(self)

    def __iter__(self):
        return
        yield

class _EmptyNode(Node, _EmptyParam):
    def __init__(self):
        Node.__init__(self)
        _EmptyParam.__init__(self)

    def newgrad(self):
        return _EmptyParam()

class Tanh(_EmptyNode):
    def __init__(self):
        super().__init__()

    def feed(self, x):
        return np.tanh(x)

    def feed_mem(self, x):
        return self.feed(x), x

    def backprop(self, grad, m, dy):
        return dy*tanh_deriv(m)

class Sigmoid(_EmptyNode):
    def __init__(self):
        super().__init__()

    def feed(self, x):
        return sigmoid(x)

    def feed_mem(self, x):
        return self.feed(x), x

    def backprop(self, grad, m, dy):
        return dy*sigmoid_deriv(m)

class Product(_EmptyNode):
    def __init__(self):
        super().__init__()

    def feed(self, x):
        return x[0]*x[1]

    def feed_mem(self, x):
        return self.feed(x), x

    def backprop(self, grad, m, dy):
        return m[1]*dy, m[0]*dy

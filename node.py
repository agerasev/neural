import numpy as np
from util import *


class Node:
    def __init__(self):
        pass
    
    def feed(self, x):
        raise NotImplementedError()
    
    def feed_mem(self, x):
        raise NotImplementedError()
    
    def newgrad(self):
        raise NotImplementedError()

    def backprop(self, grad, m, ey):
        raise NotImplementedError()
    
    def learn(self, grad, rate):
        raise NotImplementedError()

class Affine(Node):
    def __init__(self, sx, sy, mag=1e-1):
        super().__init__
        self.W = mag*np.random.randn(sx, sy)

    def feed(self, x):
        return np.dot(x, self.W)

    def feed_mem(self, x):
        return self.feed(x), x

    def newgrad(self):
        return np.zeros_like(self.W)

    def backprop(self, grad, m, ey):
        grad += np.outer(m, ey)
        return np.dot(self.W, ey)

    def learn(self, grad):
        self.W -= grad

class Bias(Node):
    def __init__(self, s, mag=1e-1):
        super().__init__()
        self.b = mag*np.random.randn(s)

    def feed(self, x):
        return self.b + x

    def feed_mem(self, x):
        return self.feed(x), None

    def newgrad(self):
        return np.zeros_like(self.b)

    def backprop(self, grad, m, ey):
        grad += ey
        return ey

    def learn(self, grad):
        self.b -= grad

class _EmptyNode(Node):
    def __init__(self):
        super().__init__()

    def newgrad(self):
        return None

    def learn(self, grad):
        pass

class Tanh(_EmptyNode):
    def __init__(self):
        super().__init__()

    def feed(self, x):
        return np.tanh(x)

    def feed_mem(self, x):
        return self.feed(x), x

    def backprop(self, grad, m, ey):
        return ey*tanh_deriv(m)

class Sigmoid(_EmptyNode):
    def __init__(self):
        super().__init__()

    def feed(self, x):
        return sigmoid(x)

    def feed_mem(self, x):
        return self.feed(x), x

    def backprop(self, grad, m, ey):
        return ey*sigmoid_deriv(m)

class Product(_EmptyNode):
    def __init__(self):
        super().__init__()

    def feed(self, x):
        return x[0]*x[1]

    def feed_mem(self, x):
        return self.feed(x), x

    def backprop(self, grad, m, ey):
        return m[1]*ey, m[0]*ey

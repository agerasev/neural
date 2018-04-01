import numpy as np

from .optim import *


class Solver:
    def __init__(self, net, loss, optims, **params):
        self.params = params
        self.name = params.get("name", "Unnamed")
        
        self.net = net
        self.loss_layer = loss
        
        if isinstance(optims, Optim):
            optims = [optims]
        self.optims = optims

    def forward(self, x, r=None):
        ny, nc = self.net.forward(x)
        y, loss, lc = self.loss_layer.forward(ny, r)
        
        if r is not None:
            for optim in self.optims:
                if hasattr(optim, "loss"):
                    loss += optim.loss(self.net)
        
        return y, loss, (nc, lc)
    
    def backward(self, cache):
        nc, lc = cache
        grad = self.net.newgrad()
        
        dy = self.loss_layer.backward(lc)
        dx = self.net.backward(grad, nc, dy)
        
        return dx, grad
    
    def learn(self, x, r):
        y, loss, cache = self.forward(x, r)
        dx, grad = self.backward(cache)

        for optim in self.optims:
            optim.learn(self.net, grad)

        return y, loss, dx
            
    def sample(self, x, r=None):
        return self.forward(x, r)[0:2]


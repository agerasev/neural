import numpy as np


def reg_loss(node, reg_rate=1e-3):
    loss = 0.0
    for W in node:
        loss += reg_rate*np.sum(W**2, axis=None)
    return loss

class Optim:
    def __init__(self, **kwargs):
        self.opts = kwargs

    def regularize(self, node, grad):
        reg_rate = self.opts.get("reg_rate", 0)
        if reg_rate != 0:
            for W, dW in zip(node, grad):
                dW += 2*reg_rate*W

    def modgrad(self, grad, **kwargs):
        norm = kwargs.get("norm", 0)
        clip = kwargs.get("clip", 0)
        for dW in grad:
            if norm != 0:
                dW /= norm
            if clip != 0:
                np.clip(dW, -clip, clip, out=dW)

    def learn(self, node, grad, **kwargs):
        raise NotImplementedError()

class SGD(Optim):
    def __init__(self, **kwargs):
        Optim.__init__(self, **kwargs)
        self.rate = kwargs["learn_rate"]

    def learn(self, node, grad, **kwargs):
        self.modgrad(grad, **kwargs)
        self.regularize(node, grad)
        for W, dW in zip(node, grad):
            W -= self.rate*dW

class Adagrad(Optim):
    def __init__(self, **kwargs):
        Optim.__init__(self, **kwargs)
        self.rate = kwargs["learn_rate"]
        self.adagrad = kwargs["adagrad"]

    def learn(self, node, grad, **kwargs):
        self.modgrad(grad, **kwargs)
        self.regularize(node, grad)
        for W, dW, aW in zip(node, grad, self.adagrad):
            aW += dW**2
            W -= self.rate*dW/np.sqrt(aW + 1e-8)

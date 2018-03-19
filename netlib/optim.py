import numpy as np


def reg_loss(node, reg_rate=1e-3):
    loss = 0.0
    for W in node:
        loss += reg_rate*np.sum(W**2, axis=None)
    return loss

def modgrad(grad, norm=1, clip=0):
    for dW in grad:
        if norm != 1 or norm != 0:
            dW /= norm
        if clip != 0:
            np.clip(dW, -clip, clip, out=dW)

def init_accum(accum, node):
    if accum is None:
        return node.newgrad()
    return accum

class Optim:
    def __init__(self):
        pass

    def learn(self, node, grad):
        raise NotImplementedError()

class Reg(Optim):
    def __init__(self, reg):
        super().__init__()
        self.reg = reg

    def learn(self, node, grad):
        if self.reg != 0:
            for W in node:
                W -= 2*self.reg*W

class SGD(Optim):
    def __init__(self, learn_rate):
        super().__init__()
        self.rate = learn_rate

    def learn(self, node, grad):
        for W, dW in zip(node, grad):
            W -= self.rate*dW

class Momentum(Optim):
    def __init__(self, learn_rate, momentum=None, frict=0.9):
        super().__init__()
        self.rate = learn_rate
        self.momentum = momentum
        self.frict = frict

    def learn(self, node, grad):
        if self.momentum is None:
            self.momentum = node.newgrad()
        for W, dW, M in zip(node, grad, self.momentum):
            M = self.frict*M + dW
            W -= self.rate*M

class Adagrad(Optim):
    def __init__(self, learn_rate, adagrad=None, decay=1, eps=1e-8):
        super().__init__()
        self.rate = learn_rate
        self.adagrad = adagrad
        self.decay = decay
        self.eps = eps

    def learn(self, node, grad):
        self.adagrad = init_accum(self.adagrad, node)
        for W, dW, A in zip(node, grad, self.adagrad):
            if self.decay == 1:
                A += dW**2
            else:
                A = self.decay*A + dW**2
            W -= self.rate*dW/np.sqrt(A + self.eps)

class Adadelta(Optim):
    def __init__(self, gamma=0.9, eps=1e-6):
        super().__init__()
        self.adagrad = None
        self.accum = None
        self.gamma = gamma
        self.eps = eps

    def learn(self, node, grad):
        self.adagrad = init_accum(self.adagrad, node)
        self.accum = init_accum(self.accum, node)
        g = self.gamma
        for W, dW, A, D in zip(node, grad, self.adagrad, self.accum):
            A = g*A + (1.0 - g)*dW**2
            updW = dW*np.sqrt((D + self.eps)/(A + self.eps))
            D = g*D + (1.0 - g)*updW**2
            W -= updW

class RMSProp(Optim):
    def __init__(self, learn_rate, accum=None, gamma=0.9):
        super().__init__()
        self.rate = learn_rate
        self.accum = accum
        self.gamma = gamma

    def learn(self, node, grad):
        self.accum = init_accum(self.accum, node)
        for W, dW, A in zip(node, grad, self.accum):
            A = self.gamma*A + (1.0 - self.gamma)*dW**2
            W -= self.rate*dW/np.sqrt(A + 1e-8)

class Adam(Optim):
    def __init__(self, learn_rate, beta1=0.9, beta2=0.999, eps=1e-8):
        super().__init__()
        self.rate = learn_rate
        self.M, self.V = None, None
        self.bM, self.bV = 1.0, 1.0
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps

    def learn(self, node, grad):
        self.M = init_accum(self.M, node)
        self.V = init_accum(self.V, node)
        for W, dW, M, V in zip(node, grad, self.M, self.V):
            M = self.beta1*M + (1.0 - self.beta1)*dW
            V = self.beta2*V + (1.0 - self.beta2)*dW**2
            self.bM *= self.beta1
            self.bV *= self.beta2
            M_ = M/(1.0 - self.bM)
            V_ = V/(1.0 - self.bV)
            W -= self.rate*M_/(np.sqrt(V_) + self.eps)

import numpy as np

from .optim import *
from .util import *


class Solver:
    def __init__(self, net, optims, **params):
        self.params = params
        self.name = params["name"]
        self.net = net
        
        if isinstance(optims, Optim):
            optims = [optims]
        self.optims = optims

        self.loss_history = []


class RNNSolver(Solver):
    def __init__(self, net, optims, **params):
        super().__init__(net, optims, **params)
        
    def learn(self, inseq, outseq):
        if inseq.shape[0] != outseq.shape[0]:
            mlen = min((inseq.shape[0], outseq.shape[0]))
            inseq = inseq[0:mlen]
            outseq = outseq[0:mlen]
        seqlen = inseq.shape[0]
        
        loss = 0.0
        cache = []
        
        h = self.net.newstate(inseq.shape[1])
        if isinstance(h, np.ndarray):
            h = (h,)
        for x, r in zip(inseq, outseq):
            ovs, m = self.net.forward((x, *h))
            y, h = ovs[0], ovs[1:]
            cache.append((m, y, r))
            loss += rms_loss(y, r)
        loss /= len(cache)

        grad = self.net.newgrad()
        dh = self.net.newstate(inseq.shape[1])
        if isinstance(dh, np.ndarray):
            dh = (dh,)
        for m, y, r in reversed(cache):
            ovs = self.net.backward(grad, m, (rms_deriv(y, r), *dh))
            dh = ovs[1:]

        modgrad(grad, norm=1, clip=5)
        for optim in self.optims:
            optim.learn(self.net, grad)

        self.loss_history.append(loss)
            
    def sample(self, seed, seqlen):
        h = self.net.newstate(1)
        if isinstance(h, np.ndarray):
            h = (h,)
        seq = []
        x = seed
        seq += seed
        for i in range(seqlen):
            ovs, _ = self.net.forward((x, *h))
            y, h = ovs[0], ovs[1:]
            seq.append(y)
            x = y
        return seq


def plot_solvers(plt, sols, red=10, win=100):
    if isinstance(sols, Solver):
        sols = [sols]
        
    plt.subplot(2, 1, 1)
    for sol in sols:
        lh = np.array(sol.loss_history)
        lh.resize(lh.shape[0]//red, red)
        plt.plot(np.mean(lh, axis=-1), label=sol.name)
    plt.legend()

    plt.subplot(2, 1, 2)
    for sol in sols:
        plt.plot(
            sol.loss_history[-win:], 
            label="%s loss: %.4f" % (sol.name, sol.loss_history[-1])
        )
    plt.legend()
    
    plt.show()

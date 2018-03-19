import numpy as np

from .layer import Layer, Param


def _wrap_list(t):
    if isinstance(t, (list, tuple)):
        return t
    else:
        return t,

def _unwrap_list(t):
    if len(t) == 0:
        return None
    elif len(t) == 1:
        return t[0]
    else:
        return t

def _wrap_link(l):
    if isinstance(l, (list, tuple)):
        return l
    else:
        return l, 0

class NetGrad(Param):
    def __init__(self, params):
        Param.__init__(self)
        self.params = params

    def __iter__(self):
        for param in self.params:
            for p in param:
                yield p

class Net(Layer, Param):
    def __init__(self, nodes, links, wmag=1):
        Layer.__init__(self)
        Param.__init__(self)

        self.nodes = nodes
        self.links = [(_wrap_link(l[0]), _wrap_link(l[1])) for l in links]

        self.ilmap = {il: [l[0] for l in self.links if l[1] == il] for il in set([l[1] for l in self.links])}
        self.olmap = {ol: [l[1] for l in self.links if l[0] == ol] for ol in set([l[0] for l in self.links])}
        
        self.incnt = {i: [] for i in range(-1, len(nodes))}
        self.oncnt = {i: [] for i in range(-1, len(nodes))}
        for ol, il in self.links:
            self.incnt[il[0]].append(il[1])
            self.oncnt[ol[0]].append(ol[1])
        for i in self.incnt:
            self.incnt[i] = sorted(set(self.incnt[i]))
        for i in self.oncnt:
            self.oncnt[i] = sorted(set(self.oncnt[i]))

        self.ios = (len(self.oncnt[-1]), len(self.incnt[-1]))

        if wmag != 1:
            for W in self:
                W *= wmag

    def __iter__(self):
        for node in self.nodes:
            for p in node:
                yield p

    def _propagate(self, grad, in_cache, v, back=False):
        v = _wrap_list(v)

        if not back:
            out_cache = [None]*len(self.nodes)
        else:
            out_cache = None

        ivs = {k: None for k in set([link[1] for link in self.links])}
        ovs = {k: None for k in set([link[0] for link in self.links])}
        used = [False]*len(self.nodes)

        if not back:
            ilmap, olmap = self.ilmap, self.olmap
            incnt, oncnt = self.incnt, self.oncnt
            iio = 0
        else:
            ivs, ovs = ovs, ivs
            ilmap, olmap = self.olmap, self.ilmap
            incnt, oncnt = self.oncnt, self.incnt
            iio = 1

        assert len(v) == self.ios[iio], "net ios count mismatch (%s != %s)" % (len(v), self.ios[iio])
        for oi in oncnt[-1]:
            ovs[(-1, oi)] = v[oi]

        while True:
            steps = 0
            for il, ols in ilmap.items():
                povs = [ovs[ol] for ol in ols]
                if ivs[il] is None and all([ov is not None for ov in povs]):
                    ivs[il] = sum(povs)
                    steps += 1

            for i, node in enumerate(self.nodes):
                if used[i]:
                    continue
                inv = [ivs[(i, ii)] for ii in incnt[i]]
                if all([v is not None for v in inv]):
                    assert len(inv) == node.ios[iio], "node(%s) ios count mismatch (%s != %s)" % (i, len(inv), node.ios[iio])
                    nx = _unwrap_list(inv)
                    if not back:
                        ny, nm = node.forward(nx)
                        out_cache[i] = nm
                    else:
                        ny = node.backward(grad.params[i], in_cache[i], nx)
                    onv = _wrap_list(ny)
                    used[i] = True
                    steps += 1
                    for j, v in enumerate(onv):
                        ovs[i, j] = v

            if steps == 0:
                break

        assert all(used), "no steps remaining while not all nodes are used, check network connectivity"

        y = _unwrap_list([ivs[(-1, ii)] for ii in incnt[-1]])
        return y, out_cache

    def forward(self, x):
        return self._propagate(None, None, x)

    def backward(self, grad, cache, dy):
        return self._propagate(grad, cache, dy, back=True)[0]

    def newgrad(self):
        return NetGrad([node.newgrad() for node in self.nodes])

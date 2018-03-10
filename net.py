from node import Node

import numpy as np


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

class Net(Node):
    def __init__(self, nodes, links):
        super().__init__()
        self.nodes = nodes
        self.links = [(_wrap_link(l[0]), _wrap_link(l[1])) for l in links]

        self.ilmap = {il: [l[0] for l in self.links if l[1] == il] for il in set([l[1] for l in self.links])}
        self.olmap = {ol: [l[1] for l in self.links if l[0] == ol] for ol in set([l[0] for l in self.links])}
        
        self.inmap = {i: [] for i in range(-1, len(nodes))}
        self.onmap = {i: [] for i in range(-1, len(nodes))}
        for ol, il in self.links:
            self.inmap[il[0]].append(il[1])
            self.onmap[ol[0]].append(ol[1])
        for i in self.inmap:
            self.inmap[i] = sorted(self.inmap[i])
        for i in self.onmap:
            self.onmap[i] = sorted(self.onmap[i])

    def _feed(self, x, mem=False):
        x = _wrap_list(x)
        if mem:
            m = [None]*len(self.nodes)
        else:
            m = None

        ivs = {k: None for k in set([link[1] for link in self.links])}
        ovs = {k: None for k in set([link[0] for link in self.links])}
        used = [False]*len(self.nodes)
        
        for oi in self.onmap[-1]:
            ovs[(-1, oi)] = x[oi]

        while not all(used):
            steps = 0
            for il, ols in self.ilmap.items():
                povs = [ovs[ol] for ol in ols]
                if all([ov is not None for ov in povs]):
                    ivs[il] = sum(povs)
                    steps += 1

            for i, node in enumerate(self.nodes):
                if used[i]:
                    continue
                inv = [ivs[(i, ii)] for ii in self.inmap[i]]
                if all([v is not None for v in inv]):
                    nx = _unwrap_list(inv)
                    if not mem:
                        nx = node.feed(nx)
                    else:
                        nx, nm = node.feed_mem(nx)
                        m[i] = nm
                    onv = _wrap_list(nx)
                    used[i] = True
                    steps += 1
                    for j, v in enumerate(onv):
                        ovs[i, j] = v

            if steps == 0:
                raise Exception("no steps while not all nodes are used, check network connectivity")

        y = _unwrap_list([ivs[(i, ii)] for ii in self.inmap[-1]])
        return y, m

    def feed(self, x):
        return self._feed(x, mem=False)[0]

    def feed_mem(self, x):
        return self._feed(x, mem=True)

    def newgrad(self):
        return [node.newgrad() for node in self.nodes]

    def backprop(self, grad, m, dy):
        dy = _wrap_list(dy)

        ivs = {k: None for k in set([link[1] for link in self.links])}
        ovs = {k: None for k in set([link[0] for link in self.links])}
        used = [False]*len(self.nodes)
        
        for ii in self.inmap[-1]:
            ivs[(-1, ii)] = dy[ii]

        while not all(used):
            steps = 0
            for ol, ils in self.olmap.items():
                pivs = [ivs[il] for il in ils]
                if all([iv is not None for iv in pivs]):
                    ovs[ol] = sum(pivs)
                    steps += 1

            for i, node in enumerate(self.nodes):
                if used[i]:
                    continue
                onv = [ovs[(i, oi)] for oi in self.onmap[i]]
                if all([v is not None for v in onv]):
                    inv = _wrap_list(node.backprop(grad[i], m[i], _unwrap_list(onv)))
                    used[i] = True
                    steps += 1
                    for j, v in enumerate(inv):
                        ivs[i, j] = v

            if steps == 0:
                raise Exception("no steps while not all nodes are used, check network connectivity")

        return _unwrap_list([ovs[(i, oi)] for oi in self.onmap[-1]])

    def _learn(self, grad):
        for node, ngrad in zip(self.nodes, grad):
            node._learn(ngrad)

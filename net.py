from node import Node

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
        self.links = [_wrap_link(l) for l in links]
        self.ilmap = {il: [l[0] for l in self.links if l[1] == il] for il in set([l[1] for l in self.links])}
        self.olmap = {ol: [l[1] for l in self.links if l[0] == ol] for ol in set([l[0] for l in self.links])}

    def feed(self, x):
        x = _wrap_list(x)

        ivs = {k: None for k in set([link[1] for link in self.links])}
        ovs = {k: None for k in set([link[0] for link in self.links])}
        used = [False]*len(self.nodes)
        
        for k in ovs:
            if k[0] == -1:
                ovs[k] = x[k[1]]

        while not all(used):
            steps = 0
            for il, ols in self.ilmap.items():
                povs = [ovs[ol] for ol in ols]
                if all([ov is not None for ov in povs]):
                    ivs[il] = sum(povs)
                    steps += 1


            for i, node in enumrate(self.nodes):
                if used[i]:
                    continue
                inm = {}
                for il, iv in ivs.items():
                    if il[0] == i:
                        inm[il[1]] = iv
                inv = [inm[k] for k in sorted(inm.keys())]
                if all([v is not None for v in inv]):
                    onv = _wrap_list(node.feed(_unwrap_list(inv)))
                    used[i] = True
                    steps += 1
                    for j, v in enumerate(onv):
                        ovs[i, j] = v

            if steps == 0:
                raise Exception("no steps while not all nodes are used, check network connectivity")

        onm = {}
        for k in ivs:
            if k[0] == -1:
                onm[k[1]] = ivs[k]
        return _unwrap_list([onm[k] for k in sorted(inm.keys())])

"""
    def feed_mem(self, x):
        return self.feed(x), x

    def newgrad(self):
        return np.zeros_like(self.W)

    def backprop(self, grad, m, ey):
        grad += np.outer(m, ey)
        return np.dot(self.W, ey)

    def learn(self, grad):
        self.W -= grad
"""

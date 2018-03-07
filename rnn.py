from .util import *
from .node import *

class RNN(Node):
    def __init__(self, sx, sh, sy, mag=1e-1):
        super().__init__(self)
        self.nodes = [
            Matrix(sx, sh, mag=mag), # W_xh
            Matrix(sh, sh, mag=mag), # W_hh
            Matrix(sh, sy, mag=mag), # W_hy
            Tanh()
        ]
    
    def newgrad(self):
        return [node.newgrad() for noe in self.nodes]
    
    def newstate(self):
        return np.zeros(self.sizes[1], dtype=np.float64)
    
    def feed(self, xh, mem=False):
        x, h = xh
        W_xh, W_hh, W_hy, th = self.nodes
        v = np.dot(x, W_xh) + np.dot(h, W_hh)
        a = np.tanh(v)
        y = np.dot(a, W_hy)
        if mem:
            m = [x, h, v, a, y]
        else:
            m = None
        return a, softmax(y), m
    
    def backprop_step(self, grad, eh, m, ey):
        W_xh, W_hh, W_hy = self.params
        dW_xh, dW_hh, dW_hy = grad
        x, h, v, a, y = m
        
        dW_hy += np.outer(a, ey)
        ea = np.dot(W_hy, ey) + eh
        ev = ea*tanh_deriv(v)
        dW_xh += np.outer(x, ev)
        dW_hh += np.outer(h, ev)
        eh = np.dot(W_hh, ev)
        
        return eh
    
    def learn(self, grad, learning_rate, adagrad=None, rmsprop=0):
        if adagrad is None: 
            for W, dW in zip(self.params, grad):
                W -= learning_rate*dW
        else:
            for W, dW, aW in zip(self.params, grad, adagrad):
                if rmsprop == 0:
                    aW += dW**2
                else:
                    aW *= 1 - rmsprop
                    aW += rmsprop*dW**2
                W -= learning_rate*dW/np.sqrt(aW + 1e-8)

class LSTM:
    def __init__(self, sx, sh, sc, sy, mag=1e-1):
        self.sizes = (sx, sh, sc, sy)
        layer_sizes = [
            (sx, sh), (sh, sc), (sh, sy), # W_xh, W_hc, W_hy
            (sh, sc), (sh, sc), (sh, sc) # W_f, W_i, W_o
        ]
        self.params = [ mag*np.random.randn(*s) for s in layer_sizes]
    
    def newgrad(self):
        return [np.zeros_like(v) for v in self.params]
    
    def newstate(self):
        return (np.zeros(self.sizes[1], dtype=np.float64), np.zeros(self.sizes[2], dtype=np.float64))
    
    def step(self, hc, x, mem=False):
        h, c = hc
        W_xh, W_hc, W_hy, W_f, W_i, W_o = self.params
        
        x_w = np.dot(x, W_xh)
        xh = x_w + h
        
        f_w = np.dot(xh, W_f)
        f_s = sigmoid(f_w)
        c_m = f_s*c
        
        i_w = np.dot(xh, W_i)
        i_s = sigmoid(i_w)
        hc_w = np.dot(xh, W_hc)
        hc_t = np.tanh(hc_w)
        hc_m = i_s*hc_t
        c_a = c_m + hc_m
        
        o_w = np.dot(xh, W_o)
        o_s = sigmoid(o_w)
        ch_t = np.tanh(c_a)
        ch_m = o_s*ch_t
        
        y = np.dot(ch_m, W_hy)
        
        if mem:
            m = [
                x, xh, c,
                f_w, f_s,
                i_w, i_s, hc_w, hc_t,
                o_w, o_s, c_a, ch_t,
                ch_m
            ]
        else:
            m = None
        return (ch_m, c_a), softmax(y), m
    
    def backprop_step(self, grad, ehc, m, ey):
        eh, ec = ehc
        W_xh, W_hc, W_hy, W_f, W_i, W_o = self.params
        dW_xh, dW_hc, dW_hy, dW_f, dW_i, dW_o = grad
        x, xh, c, f_w, f_s, i_w, i_s, hc_w, hc_t, o_w, o_s, c_a, ch_t, ch_m = m
        
        dW_hy += np.outer(ch_m, ey)
        ech_m = np.dot(W_hy, ey) + eh
        ech_t = ech_m*o_s
        ec_a = ech_t*tanh_deriv(c_a) + ec
        
        eo_s = ech_m*ch_t
        eo_w = eo_s*sigmoid_deriv(o_w)
        dW_o += np.outer(xh, eo_w)
        exh_o = np.dot(W_o, eo_w)
        
        ehc_t = ec_a*i_s
        ehc_w = ehc_t*tanh_deriv(hc_w)
        dW_hc += np.outer(xh, ehc_w)
        exh_hc = np.dot(W_hc, ehc_w)
        
        ei_s = ec_a*hc_t
        ei_w = ei_s*sigmoid_deriv(i_w)
        dW_i += np.outer(xh, ei_w)
        exh_i = np.dot(W_i, ei_w)
        
        ef_s = ec_a*c
        ef_w = ef_s*sigmoid_deriv(f_w)
        dW_f += np.outer(xh, ef_w)
        exh_f = np.dot(W_f, ef_w)
        
        ec_i = ec_a*f_s
        exh = exh_f + exh_i + exh_hc + exh_o
        
        dW_xh += np.outer(x, exh)
        
        return (exh, ec_i)
    
    def learn(self, grad, learning_rate, adagrad=None, rmsprop=0):
        if adagrad is None: 
            for W, dW in zip(self.params, grad):
                W -= learning_rate*dW
        else:
            for W, dW, aW in zip(self.params, grad, adagrad):
                aW += dW**2
                if rmsprop != 0:
                    aW *= (1 - rmsprop)
                W -= learning_rate*dW/np.sqrt(aW + 1e-8)
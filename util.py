import numpy as np


def tanh_deriv(x):
        return 1/np.cosh(x)**2

def sigmoid(x):
    return 1/(1 + np.exp(-x))

def sigmoid_deriv(x):
    ex = np.exp(-x)
    return ex/(ex + 1)**2
    
def softmax(y):
    y = y - np.max(y)
    ey = np.exp(y)
    return ey/np.sum(ey)

def ce_loss(y, ir):
    return -np.log(y[ir])

def ce_softmax_deriv(y, ir):
    dy = np.copy(y)
    dy[ir] -= 1
    return dy

def rms_loss(a, y):
    d = a - y
    return np.dot(d, d)/2

def rms_deriv(a, y):
    return a - y

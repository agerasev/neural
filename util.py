import numpy as np


def tanh_deriv(x):
        return 1/np.cosh(x)**2

def sigmoid(x):
    ex = np.exp(x)
    return ex/(ex + 1)

def sigmoid_deriv(x):
    ex = np.exp(x)
    return ex/(ex + 1)**2
    
def softmax(y):
    y = y - np.max(y)
    ey = np.exp(y)
    return ey/np.sum(ey)

def cross_entropy_loss(y, r):
    return -np.log(np.dot(y, r))

def rms(a, y):
    d = a - y
    return np.dot(d, d)/2

def rms_deriv(a, y):
    return a - y

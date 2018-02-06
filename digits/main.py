import numpy as np
import os
import gzip
import struct


# load input data

def loadlabels(filepath):
	with gzip.open(filepath, "rb") as f:
		_magic, n = struct.unpack(">ll", f.read(2*4))
		assert(_magic == 2049)
		return f.read(n)

def loadimages(filepath):
	with gzip.open(filepath, "rb") as f:
		_magic, n, sx, sy = struct.unpack(">llll", f.read(4*4))
		assert(_magic == 2051)
		images = []
		for _ in range(n):
			images.append(np.fromstring(f.read(sx*sy), dtype=np.dtype("B")).astype(np.float32)/255.0)
		return ((sx, sy), images)

def _printimg(size, img):
	(sx, sy), data = size, img
	for iy in range(sy):
		for ix in range(sx):
			print("##" if data[iy*sx + ix] > 0.5 else "  ", end="")
		print()

def load(labelsfp, imagesfp):
	labels = loadlabels(labelsfp)
	images = loadimages(imagesfp)
	assert(len(labels) == len(images[1]))
	return (images[0], list(zip(labels, images[1])))

print("loading train set")
imgsize, trainset = load("data/train-labels-idx1-ubyte.gz", "data/train-images-idx3-ubyte.gz")

print("loading test set")
_imgsize, testset = load("data/t10k-labels-idx1-ubyte.gz", "data/t10k-images-idx3-ubyte.gz")

assert(imgsize == _imgsize)


# function definition

def act(x):
	return np.tanh(x)

def act_deriv(x):
	return 1/np.cosh(x)**2

def feedforward(net, x, mem=None):
	weights, biases = net

	a = x + biases[0]
	if mem is not None:
		mem.append(a)

	for w, b in zip(weights, biases[1:]):
		a = np.dot(w, act(a)) + b
		if mem is not None:
			mem.append(a)

	return a

# cost function
def cost(a, y):
	d = a - y
	return np.dot(d, d) # mse

def cost_deriv(a, y):
	return a - y

def backprop(net, a, y, mem, neterr, rate):
	ws, bs = net
	ews, ebs = neterr

	e = cost_deriv(a, y)
	for w, b, ew, eb, a, ap in reversed(list(zip(ws, bs[1:], ews, ebs[1:], mem[1:], mem[:-1]))):
		eb -= e*rate
		ew = np.dot(act_deriv(a).transpose(), act(ap))*rate
		e = # continue




ni, nh, no = imgsize[0]*imgsize[1], 15, 10 # input, hidden and output layers sizes
mri = 0.01 # magnitude of initial random values

# weights and biases
net = (
	[mri*np.random.randn(sx, sy) for sx, sy in zip(sizes[:-1], sizes[1:])],
	[mri*np.random.randn(s) for s in sizes]
)

batchsize = 10
rate = 1e-2
for digit, img in testset:
	out = feedforward(net, img)
	print(out.argmax(), digit)
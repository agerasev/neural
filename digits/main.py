import numpy as np
import os
import gzip
import struct
import random


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


# classes and function definition

class Net:
	def __init__(self, sizes, mag=0):
		self.sizes = sizes
		if mag == 0:
			self.weights = list([np.zeros((sx, sy), dtype=np.float32) for sx, sy in zip(sizes[1:], sizes[:-1])])
			self.biases = list([np.zeros(s, dtype=np.float32) for s in sizes])
		else:
			self.weights = list([mag*np.random.randn(sx, sy) for sx, sy in zip(sizes[1:], sizes[:-1])])
			self.biases = list([mag*np.random.randn(s) for s in sizes])

	def __iter__(self):
		return iter(self.weights + self.biases)

def act(x):
	return np.tanh(x)

def act_deriv(x):
	return 1/np.cosh(x)**2

def feedforward(net, x, fmem=False):
	a = x + net.biases[0]
	if fmem:
		mem = list([[None]*2 for _ in net.biases])
		mem[0][0] = a
	else:
		mem = None

	for i, (w, b) in enumerate(zip(net.weights, net.biases[1:])):
		a = act(a)
		if fmem:
			mem[i][1] = a

		a = np.dot(w, a) + b
		if fmem:
			mem[i + 1][0] = a

	return (a, mem)

# cost function
def cost(a, y):
	d = a - y
	return np.dot(d, d) # mse

def cost_deriv(a, y):
	return a - y

def backprop(net, a, y, mem, neterr, rate):
	e = cost_deriv(a, y)
	zmem = list(zip(*mem))
	zlist = list(zip(
		net.weights, net.biases[1:],
		neterr.weights, neterr.biases[1:],
		zmem[0][1:], zmem[1][:-1]
	))
	for w, b, ew, eb, v, ap in reversed(zlist):
		eb += e*rate
		e = e*act_deriv(v)
		ew += np.outer(e, ap)*rate
		e = np.dot(e.transpose(), w)
	neterr.biases[0] += e

net = Net((imgsize[0]*imgsize[1], 15, 10), mag=1e-2)

batchsize = 10
rate = 1e-2
for iepoch in range(1):
	print("epoch %s:" % iepoch)
	totalcost = 0.0
	for batch in [trainset[p:p+batchsize] for p in range(0, len(trainset), batchsize)]:
		neterr = Net(net.sizes)
		for digit, img in batch:
			res = np.array([i == digit for i in range(10)])
			out, mem = feedforward(net, img, fmem=True)
			totalcost += cost(out, res)
			backprop(net, out, res, mem, neterr, rate)
		for v, ev in zip(net, neterr):
			v -= ev/batchsize
	print("train cost avg: %s" % (totalcost/len(testset)))

	totalcost = 0.0
	hitcount = 0
	for digit, img in testset:
		res = np.array([i == digit for i in range(10)])
		out, _ = feedforward(net, img)
		totalcost += cost(out, res)
		hitcount += digit == np.argmax(out)
	print("test cost avg: %s" % (totalcost/len(testset)))
	print("hit count: %s" % hitcount)

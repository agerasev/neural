import numpy as np
import os
import gzip
import struct
import random
import time

_version = 0.1

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
			self.weights = [np.zeros((sx, sy), dtype=np.float32) for sx, sy in zip(sizes[1:], sizes[:-1])]
			self.biases = [np.zeros(s, dtype=np.float32) for s in sizes]
		else:
			self.weights = [mag*np.random.randn(sx, sy) for sx, sy in zip(sizes[1:], sizes[:-1])]
			self.biases = [mag*np.random.randn(s) for s in sizes]

	def __iter__(self):
		return iter(self.weights + self.biases)

	def generr(self):
		return Net(self.sizes)

	def genmem(self):
		return [[np.zeros_like(b) for _ in range(2)] for b in self.biases]

	def fill(self, value):
		for w in self.weights:
			w.fill(value)
		for b in self.biases:
			b.fill(value)

def act(x, out=None):
	return np.tanh(x, out=out)

def act_deriv(x, out=None):
	if out is not None:
		np.cosh(x, out=out)
		np.reciprocal(out, out=out)
		out **= 2
		return out
	else:
		return 1/np.cosh(x)**2

def feedforward(net, x, mem=None):
	if mem:
		m = mem[0][0]
		np.copyto(m, x)
		m += net.biases[0]
		for i, (w, b) in enumerate(zip(net.weights, net.biases[1:])):
			mx = mem[i][1]
			act(m, out=mx)
			m, mx = mx, mem[i + 1][0]
			np.dot(w, m, out=mx)
			mx += b
			m = mx
		return m
	else:
		a = x + net.biases[0]
		for w, b in zip(net.weights, net.biases[1:]):
			a = np.dot(w, act(a)) + b
			if mem:
				np.copyto(mem[i + 1][0], a)
		return a

# cost function
def cost(a, y):
	d = a - y
	return np.dot(d, d)/2

def cost_deriv(a, y, out=None):
	if out is not None:
		np.copyto(a, out)
		out -= y
	else:
		return a - y

def backprop(net, a, y, mem, err, rate):
	if True:
		e = mem[-1][1]
		cost_deriv(a, y, out=e)
		zmem = list(zip(*mem))
		zlist = list(zip(
			net.weights, net.biases[1:],
			err.weights, err.biases[1:],
			zmem[0][1:], zmem[1][:-1]
		))
		for w, b, ew, eb, mi, mpa in reversed(zlist):
			np.copyto(eb, e)
			eb *= rate
			act_deriv(mi, out=mi)
			mi *= e
			e = mi
			np.outer(e, mpa, out=ew)
			ew *= rate
			np.dot(e.transpose(), w, out=mpa)
			e = mpa
		err.biases[0] += e
	else:
		e = cost_deriv(a, y)
		zmem = list(zip(*mem))
		zlist = list(zip(
			net.weights, net.biases[1:],
			err.weights, err.biases[1:],
			zmem[0][1:], zmem[1][:-1]
		))
		for w, b, ew, eb, mi, mpa in reversed(zlist):
			eb += e*rate
			e = e*act_deriv(mi)
			ew += np.outer(e, mpa)*rate
			e = np.dot(e.transpose(), w)
		err.biases[0] += e

net = Net((imgsize[0]*imgsize[1], 15, 10), mag=1e-2)
neterr = net.generr()
netmem = net.genmem()
oneerr = net.generr()

batchsize = 10
rate = 1e-2
for iepoch in range(1):
	print("epoch %d:" % iepoch)
	print("train:")
	totalcost = 0.0
	tstart = time.time()
	for batch in [trainset[p:p+batchsize] for p in range(0, len(trainset), batchsize)]:
		for digit, img in batch:
			res = np.array([i == digit for i in range(10)])
			out = feedforward(net, img, mem=netmem)
			totalcost += cost(out, res)
			backprop(net, out, res, netmem, oneerr, rate)
			for ne, oe in zip(neterr, oneerr):
				ne += oe
		for v, ev in zip(net, neterr):
			v -= ev/batchsize
		neterr.fill(0)
	print("cost avg: %.4f" % (totalcost/len(testset)))
	print("time elapsed: %.2f s" % (time.time() - tstart))

	print("test:")
	totalcost = 0.0
	hitcount = 0
	tstart = time.time()
	for digit, img in testset:
		res = np.array([i == digit for i in range(10)])
		out = feedforward(net, img, mem=netmem)
		totalcost += cost(out, res)
		hitcount += digit == np.argmax(out)
	print("cost avg: %.4f" % (totalcost/len(testset)))
	print("hit count: %d / %d" % (hitcount, len(testset)))
	print("time elapsed: %.2f s" % (time.time() - tstart))

print("done")

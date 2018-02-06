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

def load(labelsfp, imagesfp):
	labels = loadlabels(labelsfp)
	images = loadimages(imagesfp)
	assert(len(labels) == len(images[1]))
	return (images[0], list(zip(labels, images[1])))

print("loading train set")
trainset = load("data/train-labels-idx1-ubyte.gz", "data/train-images-idx3-ubyte.gz")

print("loading test set")
testset = load("data/t10k-labels-idx1-ubyte.gz", "data/t10k-images-idx3-ubyte.gz")

def _printimg(size, img):
	(sx, sy), data = size, img
	for iy in range(sy):
		for ix in range(sx):
			print("##" if data[iy*sx + ix] > 0.5 else "  ", end="")
		print()

print(testset[1][0][0])
_printimg(testset[0], testset[1][0][1])

def sigmoid(z):
	return 1.0/(1.0 + np.exp(-z))

ni, nh, no = 25*52, 10, 10 # input, hidden and output layers sizes
mri = 0.01 # magnitude of initial random values

# weights and biases
Wih, Who = mri*np.random.randn(ni, nh), mri*np.random.randn(nh, no)
bi, bh, bo = (mri*np.random.randn(n) for n in (ni, nh, no))



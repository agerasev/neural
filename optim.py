import numpy as np


class Optim:
	def __init__(self):
		pass

	def learn(self, node, grad, **kwargs):
		raise NotImplementedError()

class SGD:
	def __init__(self, learning_rate):
		self.rate = learning_rate

	def learn(self, node, grad, **kwargs):
		for W, dW in zip(node, grad):
			rate = self.rate
			rate /= kwargs.get("batch_size", 1)
			W -= rate*dW

class Adagrad:
	def __init__(self, learning_rate, adagrad):
		self.rate = learning_rate
		self.adagrad = adagrad

	def learn(self, node, grad, **kwargs):
		for W, dW, aW in zip(node, grad, self.adagrad):
			aW += dW**2
			rate = self.rate
			rate /= kwargs.get("batch_size", 1)
			W -= rate*dW/np.sqrt(aW + 1e-8)

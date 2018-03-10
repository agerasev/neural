import numpy as np


def forgrad(grad, func, fend=lambda x: isinstance(x, np.ndarray)):
	if not fend(grad):
		for g in grad:
			forgrad(g, func, fend)
	else:
		func(grad)
		

class Optim:
	def __init__(self):
		pass

	def learn(node, grad):
		raise NotImplementedError()

class SGD:
	def __init__(self, learning_rate):
		self.rate = learning_rate

	def learn(node, grad):
		forgrad(grad, lambda x: x *= self.rate)
		node._learn(grad)

class Adagrad:
	def __init__(self, learning_rate, adagrad):
		self.rate = learning_rate
		forgrad(adagrad, lambda x: x += 1e-8)
		self.adagrad = adagrad

	def _update(ada, grad):
		ada += grad**2
		grad *= learning_rate/np.sqrt(ada)

	def learn(node, grad):
		forgrad(grad, fend=lambda x: isinstance(x[0], np.ndarray))

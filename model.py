import numpy as np

class BayesianNet:

	def __init__(self, sigma=0.1):
		self.sigma = sigma
		self.layers = []

	def add_layers(self, layers):
		new_layers = list(layers)
		self.layers += new_layers

	def get_parameters(self):
		params = []
		for layer in self.layers:
			params.extend(layer.get_parameters())
		return np.array(params)

	def compile(self, input_size):
		self.layers[0].build(input_size)
		for i in range(1, len(self.layers)):
			self.layers[i].build(self.layers[i-1].units)

	def predict(self, x):
		try:
			assert len(x[0]) == self.layers[0].input_size
		except TypeError:
			assert self.layers[0].input_size == 1
		samples = x.shape[0]
		output = np.zeros(samples)
		for i in range(samples):
			result = self.layers[0].call(x[i])
			for l in self.layers[1:]:
				result = l.call(result)
			output[i] = result
		return output

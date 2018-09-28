import activations
import numpy as np


class Neuron:

	def __init__(self, w, b, activation):
		self.w = w
		self.b = b
		self.activation = activation

	def fire(self, x):
		return self.activation(np.dot(self.w, x) + self.b)


class Layer:

	def __init__(self, units, activation):
		self.units = units
		self.activation = getattr(activations, activation)
		self.neurons = []
		self.input_size = 1
		
	def build(self, input_size):
		weights = np.random.rand(self.units, input_size)
		biases = np.random.rand(self.units)
		self.neurons = [Neuron(weights[i], biases[i], self.activation) 
							for i in range(self.units)]
		self.input_size = input_size

	def call(self, input):
		return np.array([n.fire(input) for n in self.neurons])

	def print_parameters(self):
		for i in range(self.units):
			print("Unit " + str(i) + ":")
			print("W = " + str(((self.neurons)[i]).w))
			print("b = " + str(((self.neurons)[i]).b))

	def get_parameters(self):
		params = []
		for n in self.neurons:
			params.extend([n.w, n.b])
		return params


class GaussianPrior(Layer):

	def __init__(self, units, activation, mean=0, var=1):
		super().__init__(units, activation)
		self.mu = mean
		self.sigma = var

	def build(self, input_size):
		weights = np.random.normal(self.mu, self.sigma, 
								  (self.units, input_size))
		biases = np.random.normal(self.mu, self.sigma, self.units)
		self.neurons = [Neuron(weights[i], biases[i], self.activation) 
							for i in range(self.units)]
		self.input_size = input_size




# l = GaussianPrior(10, activation='relu', mean=0, var=1)
# l.build(2)
# l.print_parameters()
# result = l.call(np.array([2,3]))
# print(result)
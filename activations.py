import autograd.numpy as ag_np
import numpy as np


def tanh(x):
	return ag_np.tanh(x)

def relu(x):
	return np.maximum(0, x)

def rbf(x):
	return ag_np.exp(-1*ag_np.square(x))

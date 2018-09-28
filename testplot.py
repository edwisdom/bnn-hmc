import matplotlib.pyplot as plt 
import numpy as np


x = np.linspace(-20, 20, 200)
ys = np.random.rand(4, 200)
plt.figure()
for y in ys:
	plt.plot(x, y, '.-')
plt.show()
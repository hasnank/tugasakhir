import numpy as np

for i in range(5):
	np.random.seed(1)
	for j in range(5):
		print(np.random.rand(1))
		print(np.random.randint(1,5))
		
# Write a Python function that performs linear regression using gradient descent. 

# Input:
# X = np.array([[1, 1], [1, 2], [1, 3]]), y = np.array([1, 2, 3]), alpha = 0.01, iterations = 1000
# Output:
# np.array([0.1107, 0.9513])

import numpy as np
def linear_regression_gradient_descent(X: np.ndarray, y: np.ndarray, alpha: float, iterations: int) -> np.ndarray:
	X = X[:, 1]
	w=0.0
	b=0.0
	for i in range(iterations):
		y_pred = np.dot(X, w) + b
		error=(y_pred-y)
		gradient=np.dot(X, error)/len(X)
		gradientb=(np.sum(error))/len(X)
		w=w - alpha*gradient
		b-= alpha*gradientb
		if(i%100==0):
			loss=(y_pred-y)**2
			loss=loss.sum()/len(X)
			print(f"after iteration {i} loss=",loss)
	return  round(b, 4), round(w, 4)

X = np.array([[1, 1], [1, 2], [1, 3]])
y = np.array([1, 2, 3])
alpha = 0.01
iterations = 1000

print(linear_regression_gradient_descent(X, y, alpha, iterations))
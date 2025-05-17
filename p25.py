import numpy as np
features = [[1.0, 2.0], [2.0, 1.0], [-1.0, -2.0]]
labels = [1, 0, 0]
initial_weights = [0.1, -0.2]
initial_bias = 0.0
learning_rate = 0.1
epochs = 2
def train_neuron(features: np.ndarray, labels: np.ndarray, initial_weights: np.ndarray, initial_bias: float, learning_rate: float, epochs: int) -> (np.ndarray, float, list[float]):
	features=np.array(features)
	labels=np.array(labels)
	initial_weights=np.array(initial_weights)
	def sigmoid(x):
		return 1/( 1 + np.exp(-x))
	prediction=np.dot(features, initial_weights) + initial_bias
	final=sigmoid(prediction)
	print("Initial prediction:", np.round(final,4))
	for i in range(epochs):
		mse=(labels-final)**2
	print(mse)
	
train_neuron(features, labels, initial_weights, initial_bias, learning_rate, epochs)
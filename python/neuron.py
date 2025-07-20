import numpy as np

# calc the probability of neuron
# X : the back value
# W : the weight of the back info
# b : 

def model(X, W, b):
	Z = X.dot(W) + b
	A = 1 / (1 + np.exp(-Z))
	return (A)


# Calc the cout of the neuron

def cout(A, y):
	return 1/len(y) * np.sum(-y * np.log(A) - (1 - y) * np.log(1 - A))


#

def gradients(A, X, y):
	dW = 1 / len(y) * np.dot(X.T, A - y)
	db = 1 / len(y) * np.sum(A - y)
	return (dW, db)


# change the value of the weight

def update(dW, db, W, b, learning_rate):
	W = W - learning_rate * dW
	b = b - learning_rate * db
	return (W, b)


def artificial_neuron(X, y, W, b, learning_rate = 0.1, n_iteration = 100):
	history = []
	Loss = []
	
	for i in range(n_iteration):
		A = model(X, W, b)
		Loss.append(cout(A, y))
		dW, db = gradients(A, X, y)
		W, b = update(dW, db, W, b, learning_rate)
		history.append([W, b, Loss, i])
	return (W, b, Loss)

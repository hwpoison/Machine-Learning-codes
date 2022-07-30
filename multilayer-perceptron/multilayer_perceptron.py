from email.policy import default
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split

# load iris
from sklearn.datasets import load_iris

__autor__ = 'hwpoison'


class NeuralLayer():
	def __init__(self, inputs_length, neurons_amount):
		self.synaptic_weights = np.random.randn(inputs_length, neurons_amount) * np.sqrt(2/inputs_length)
		self.bias = [-1 for i in range(neurons_amount)]

class NeuralNetwork():
	def __init__(self, layer_model):
		self.layers = layer_model
		self.learn_rate = 0.01
		self.results = {'MSE':[]}
		self.default_activation = 'tanh'
		if self.default_activation == 'tanh':
			self.derivation_function = self.tanh_derivate
			self.activation_function = self.tanh
		else:
			self.derivation_function = self.sigmoid_derivate
			self.activation_function = self.sigmoid

	def tanh(self, x):
		return np.tanh(x)
	
	def tanh_derivate(self, x):
		return 1.-np.tanh(x)**2 # sech^2{x}
		
	def sigmoid(self, x):
		return 1 / (1 + np.exp(-x))

	def sigmoid_derivate(self, x):
		return x * (1 - x)

	def backpropagation(self, input_data, output_data):
		# Initializer Back Propagation
		network_outputs = self.forward(input_data) # all layers outputs from left to right
		layer_error = (output_data - network_outputs[-1])
		# Iterate to backward from out to in
		for layer in reversed(self.layers):
			n_layer = self.layers.index(layer) # index of current layer
			delta = layer_error * self.derivation_function(network_outputs[n_layer])
			if(n_layer==0): # output layer
				gradient = input_data.T.dot(delta)
			else: # for hidden layers
				gradient = network_outputs[n_layer-1].T.dot(delta)
			layer_error = delta.dot(self.layers[n_layer].synaptic_weights.T) 
			
			# Gradient descendent, Updates weights and bias
			layer.synaptic_weights += gradient * self.learn_rate
			layer.bias += delta.sum(axis=0) * self.learn_rate

	def train(self, training_set_inputs,
			  training_set_outputs,
			  epoch_number=False):
		epochs_control = 0
		while True:
			self.backpropagation(training_set_inputs, training_set_outputs)
			network_output = self.input(training_set_inputs)
			# Mean Square Error
			MSE = np.mean(np.square(training_set_outputs - network_output))
			self.results['MSE'].append(MSE)
			epochs_control += 1
			# auto stop
			if MSE < 0.0001 or epochs_control == epoch_number:
				break

	def forward(self, input):
		output_stack = []
		for layer in self.layers:
			# Softmax for output layer
			dot = np.dot(input, layer.synaptic_weights) + layer.bias
			output = self.activation_function(dot)
			output_stack.append(output)
			input = output
		return output_stack

	def input(self, input):
		return self.forward(input)[-1] # foward net and return output

	def show_results(self):
		plt.plot(self.results['MSE'])
		plt.xlabel('MSE')
		plt.legend(['MSE'], loc='upper left')

		plt.show()


def one_hot(x, k, dtype=np.float32):
    """Create a one-hot encoding of x of size k."""
    return np.array(x[:, None] == np.arange(k), dtype)

if __name__ == '__main__':
	# Load Iris dataset
	iris = load_iris()
	X = iris.data
	y = iris.target
	y = one_hot(y, 3)

	# Split dataset
	X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=20, random_state=4)

	# Create Neural Network
	neural_network = NeuralNetwork([
		NeuralLayer(4, 8),  
		NeuralLayer(8, 3)
	])	
	neural_network.default = 'tanh'
	neural_network.learn_rate = 0.0001

	# Train Neural Network
	neural_network.train(X_train, y_train, epoch_number=12900)

	# Test Neural Network
	out = neural_network.input(X_test)
	corrects, wrongs = 0, 0
	for sample_n in range(len(y_test)):
		if np.argmax(y_test[sample_n]) == np.argmax(out[sample_n]):
			corrects += 1
		else:
			wrongs += 1

	print('Corrects: ', corrects)
	print('Wrongs: ', wrongs)
	print("Accuracy: ", corrects/(corrects+wrongs))

	neural_network.show_results()

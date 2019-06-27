#!/usr/bin/env python
from __future__ import print_function
import numpy as np

__autor__ = 'srbill1996'


class Perceptron():

    def __init__(self,
                 input_length=4,
                 learning_rate=0.01,
                 bias=1.0):
        super(Perceptron, self).__init__()
        self.input_length = input_length
        self.learning_rate = learning_rate
        self.bias = bias
        self.synapse_weights = []

    def zigma(self, x):
        # Input values into perceptron and generate a output
        z = np.dot(x, self.synapse_weights) + self.bias
        return z

    def predict(self, x):
        return self.activation(x)

    def activation(self, x):
        # Activation function type step
        return 1 if self.zigma(x) > 1 else 0

    def train(self, X_data, y_data, epochs=None):
        """
            X_data: input data
            y_data: expected data
            epochs: number of cycles
        """
        # Initialize weights
        self.synapse_weights = np.random.rand(self.input_length)
        epoch_count = 0  # for specific number of epochs
        # Initialize train
        while True:
            count_error = 0
            for x_input, y_expect in zip(X_data, y_data):
                output = self.predict(x_input)
                error = y_expect - output
                if(output != y_expect):
                    # if output != y_expect output, update weights
                    count_error += 1
                    update_value = self.learning_rate * error * x_input
                    self.synapse_weights += update_value

            epoch_count += 1
            # if there are no more errors why keep adjusting?
            if count_error == 0 or epoch_count == epochs:
                print(f"Trained in {epoch_count} epochs")
                break


if __name__ == '__main__':
    # dataset test example
    input_data = np.array([[0, 0, 1, 0],
                           [1, 1, 1, 0],
                           [1, 0, 1, 1],
                           [0, 1, 1, 1],
                           [0, 1, 0, 1],
                           [1, 1, 1, 1],
                           [0, 0, 0, 0]])

    expect_data = np.array([0, 1, 1, 0, 0, 1, 0]).T
    # instance perceptron
    perceptron = Perceptron()

    # train
    perceptron.train(input_data, expect_data)

    # test
    for ts_input, expected in zip(input_data, expect_data):
        output = perceptron.predict(ts_input)
        expected = 'OK' if expected == output else 'FAIL'
        print(f'Input:{ts_input} Output: {output} = {expected}')

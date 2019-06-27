#!/usr/bin/env python
from __future__ import print_function
import random

__autor__ = 'srbill1996'


class Perceptron():

    def __init__(self):
        super(Perceptron, self).__init__()
        self.input_length = 4
        self.bias = 1.0
        self.learning_rate = 0.01
        self.synapse_weights = [random.uniform(
            0.001, 0.001) for i in range(0, self.input_length)]

    def dot(self, a, b):
        # zigma
        return sum(x*z for x, z in zip(a, b))

    def forward(self, inputs):
        # Input values into perceptron and generate a output
        z = self.dot(inputs, self.synapse_weights) + self.bias
        return z

    def predict(self, inputs):
        result = self.forward(inputs)
        return self.activation_function(result)

    def activation_function(self, input):
        # Activation function type step
        return 1 if input > 1 else 0

    def train(self, ts_inputs, ts_outputs, epochs=None):
        # Learn dataset
        epoch_count = 0  # for specific number of epochs
        while True:
            count_error = 0
            for ts_input, expected in zip(ts_inputs, ts_outputs):
                output = self.predict(ts_input)
                error = expected - output
                if(output != expected):
                    count_error += 1
                    for w in range(self.input_length):
                        update_value = self.learning_rate * error * ts_input[w]
                        self.synapse_weights[w] += update_value

            epoch_count += 1
            # if there are no more errors why keep adjusting?
            if count_error == 0 or epoch_count == epochs:
                print(f"Trained in {epoch_count} epochs")
                break


if __name__ == '__main__':
    # dataset test example
    ts_inputs = [[0, 0, 1, 0],
                 [1, 1, 1, 0],
                 [1, 0, 1, 1],
                 [0, 1, 1, 1],
                 [0, 1, 0, 1],
                 [1, 1, 1, 1],
                 [0, 0, 0, 0]]

    ts_outputs = [0, 1, 1, 0, 0, 1, 0]
    # instance perceptron
    perceptron = Perceptron()

    # train
    perceptron.train(ts_inputs, ts_outputs)

    # test
    for ts_input, expected in zip(ts_inputs, ts_outputs):
        output = perceptron.predict(ts_input)
        expected = 'OK' if expected == output else 'FAIL'
        print(f'Input:{ts_input} Output: {output} = {expected}')

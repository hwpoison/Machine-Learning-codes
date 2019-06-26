#!/usr/bin/env python
import numpy as np

__autor__ = 'srbill1996'


class Perceptron():

    def __init__(self):
        super(Perceptron, self).__init__()
        self.input_length = 4
        self.bias = 1.0
        self.learning_rate = 0.01
        self.synapse_weights = np.random.rand(self.input_length)

    def forward(self, inputs):
        # Input values into perceptron and generate a output
        z = np.dot(inputs, self.synapse_weights) + self.bias
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
                    # if output != expected output, update weights
                    count_error += 1
                    update_value = self.learning_rate * error * ts_input
                    self.synapse_weights += update_value

            epoch_count += 1
            # if there are no more errors why keep adjusting?
            if count_error == 0 or epoch_count == epochs:
                print(f"Trained in {epoch_count} epochs")
                break


if __name__ == '__main__':
    # dataset test example
    ts_inputs = np.array([[0, 0, 1, 0],
                          [1, 1, 1, 0],
                          [1, 0, 1, 1],
                          [0, 1, 1, 1],
                          [0, 1, 0, 1],
                          [1, 1, 1, 1],
                          [0, 0, 0, 0]])

    ts_outputs = np.array([0, 1, 1, 0, 0, 1, 0]).T
    # instance perceptron
    perceptron = Perceptron()

    # train
    perceptron.train(ts_inputs, ts_outputs)

    # test
    for ts_input, expected in zip(ts_inputs, ts_outputs):
        output = perceptron.predict(ts_input)
        expected = 'OK' if expected == output else 'FAIL'
        print(f'Input:{ts_input} Output: {output} = {expected}')

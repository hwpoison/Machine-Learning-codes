import numpy as np
import matplotlib.pyplot as plt

__autor__ = 'srbill1996'


class NeuronLayer():
    def __init__(self, inputs_length, neurons_amount):
        self.synaptic_weights = 2 * \
            np.random.random((inputs_length, neurons_amount)) - 1
        self.bias = [0 for i in range(neurons_amount)]


class NeuralNetwork():
    def __init__(self, layer_model):
        self.layers = layer_model
        self.learn_rate = 0.5
        self.errors = []

    def sigmoid_function(self, x):
        return 1 / (1 + np.exp(-x))

    def sigmoid_derivate(self, x):
        return x * (1 - x)

    def activation(self, x):
        return 1 if self.sigmoid_function(x) > 0.0 else 0

    def backpropagation(self, input_data, output_data):
        # Back Propagation
        network_outputs = self.forward(input_data)
        outputs = [input_data]
        for out in network_outputs:
            outputs.append(out)
        layer_error = output_data - network_outputs[-1]
        for i in range(len(self.layers) - 1, -1, -1):
            delta = layer_error * self.sigmoid_derivate(outputs[i+1])
            layer_error = delta.dot(self.layers[i].synaptic_weights.T)
            gradient = outputs[i].T.dot(delta)
            # gradient descendent
            self.layers[i].synaptic_weights += gradient * self.learn_rate
           # print(self.layers[i].bias, delta)
            self.layers[i].bias += delta.sum(axis=0) * self.learn_rate

    def train(self, training_set_inputs,
              training_set_outputs,
              epoch_number=False):
        epochs_control = 0
        while True:
            self.backpropagation(training_set_inputs, training_set_outputs)
            network_output = self.input(training_set_inputs)
            MSE = ((training_set_outputs - network_output) ** 2).sum()
            self.errors.append(MSE)
            # if is supervised, automatic stop on min MSE
            if(epoch_number is False):
                if MSE < 0.009:
                    return True
            else:
                if(epochs_control == epoch_number):
                    return True
            epochs_control += 1

    def forward(self, input):
        output_stack = []
        for layer in self.layers:
            z = np.dot(input, layer.synaptic_weights) + layer.bias[0]
            out = self.sigmoid_function(z)
            output_stack.append(out)
            input = out
        return output_stack

    def input(self, input):
        return self.forward(input)[-1]

    def argmax(self, input):
        return np.argmax(self.input(input))

    def show_error(self):
        # Imprime los errores
        plt.plot(self.errors)
        plt.show()


if __name__ == "__main__":

    # Capa 1 = capa con 2 entradas a 5 neuronas conectadas a 1 neurona
    layer_model = [
        NeuronLayer(2, 5),
        NeuronLayer(5, 1)
    ]

    # Se asigna el modelo de capas a la red
    neural_network = NeuralNetwork(layer_model)

    def xor_problem():
        # definicion del set de entrenamiento problema XOR
        inputs = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
        outputs = np.array([[0],  [1],   [1],   [0]])

        # entrenamiento
        neural_network.train(inputs, outputs, epoch_number=1200)
        neural_network.learn_rate = 0.01
        # test
        print(neural_network.input([0, 1]))  # se espera 0.9
        neural_network.show_error()

    xor_problem()

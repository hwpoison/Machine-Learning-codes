#!/usr/bin/env python
import time
import numpy as np
import matplotlib.pyplot as plt
from random import choice, randint, shuffle
__autor__ = 'srbill1996'


class Adaline():

    def __init__(self,
                 input_length=4,
                 learning_rate=.001,
                 bias=0):
        super(Adaline, self).__init__()
        self.input_length = input_length
        self.learning_rate = learning_rate
        self.bias = bias
        self.synapse_weights = None
        self.all_errors = []

    def forward(self, inputs):
        return np.dot(inputs, self.synapse_weights) + self.bias

    def predict(self, inputs):
        return np.where(self.forward(inputs) < 0.0, 0, 1)

    def activation(self, inputs):
        return self.predict(inputs)

    def train(self, input_data, expected_output, epochs=10):
        # Se inicializan los pesos
        self.synapse_weights = np.random.uniform(
            high=0.001, size=(self.input_length, 1))
        expected_output = np.where(expected_output == 0, -1., 1.)
        # Entrenamiento por gradiente descendente estocastica
        time_init = time.time()
        for epoch in range(epochs):
            # Se elige un lote y se desordena
            batch = [[i] for i in range(0, len(expected_output)-1)]
            shuffle(batch)
            for element_index in batch:
                output = self.forward(input_data[element_index])
                # Se calcular el error
                error = expected_output[element_index] - output
                # Se calcula la gradiente descendente
                update_value = self.learning_rate * \
                    input_data[element_index].T.dot(error)
                # Se actualizan los pesos y el sesgo
                self.synapse_weights += update_value
                self.bias += self.learning_rate * error.sum()
            # Se calcula el error cuadratico medio / función de coste
            cost = expected_output - self.forward(input_data).T[0]
            sum_cost = (cost ** 2).sum() / 2.0
            self.all_errors.append(sum_cost)
        time_final = time.time() - time_init
        print(f"Trained in:{ time_final } seconds and {epochs} epochs.\nMin Error: {sum_cost}")

    def show_error(self):
        # Imprime los errores
        plt.plot(self.all_errors)
        plt.show()


if __name__ == '__main__':
    # Dataset de prueba
    ts_inputs = np.array([[0.9, 0.1, 0.7, 0.2],

                          [0.0, 0.9, 0.3, 0.8],

                          [0.2, 1.0, 0.0, 0.7],

                          [1.0, 0.0, 0.8, 0.1],

                          [0.3, 0.4, 0.3, 0.6]])

    ts_outputs = np.array([0, 1, 1, 0, 1])

    # Inicialización del modelo
    adaline = Adaline(input_length=4, learning_rate=.01)

    # Entrenamiento
    adaline.train(ts_inputs, ts_outputs, 5)

    # Testear modelo
    test_dataset = list(zip(ts_inputs, ts_outputs))
    shuffle(test_dataset)
    for ts_input, expected in test_dataset:
        output = adaline.predict(ts_input)
        expected = 'OK' if expected == output else 'FAIL'
        print(f'Input:{ts_input} Output: {output} = {expected}')

    # Mostrar error
    adaline.show_error()

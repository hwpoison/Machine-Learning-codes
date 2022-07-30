#!/usr/bin/env python
from __future__ import print_function
import matplotlib.pyplot as plt
from random import shuffle
import numpy as np
import time

__autor__ = 'hwpoison'


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

    def forward(self, x):
        return np.dot(x, self.synapse_weights) + self.bias

    def predict(self, x):
        return self.activation(x)

    def activation(self, x):
        return np.where(self.forward(x) < 0.0, 0, 1)

    def train(self, X_data, y_data, epochs=10):
        """
            X_data: input data
            y_data: expected data
            epochs: number of cycles
        """
        # Se inicializan los pesos
        self.synapse_weights = np.random.uniform(
            high=0.001, size=(self.input_length, 1))
        y_data = np.where(y_data == 0, -1., 1.)
        # Entrenamiento por gradiente descendente estocastica
        time_init = time.time()
        for epoch in range(epochs):
            # Se elige un lote y se desordena
            batch = [[i] for i in range(0, len(y_data)-1)]
            shuffle(batch)
            for element_index in batch:
                output = self.forward(X_data[element_index])
                # Se calcular el error
                error = y_data[element_index] - output
                update_value = self.learning_rate * \
                    X_data[element_index].T.dot(error)
                # Se actualizan los pesos y el sesgo
                self.synapse_weights += update_value
                self.bias += self.learning_rate * error.sum()
                # Se calcula el error cuadratico medio / función de coste
                MSE = (y_data - self.forward(X_data).T[0] ** 2).sum()
                self.all_errors.append(MSE)
        time_final = time.time() - time_init
        print(f"Trained in:{ time_final } seconds and {epochs} epochs.\nMin Error: {MSE}")

    def show_error(self):
        # Imprime los errores
        plt.plot(self.all_errors)
        plt.show()


if __name__ == '__main__':
    # Dataset de prueba
    input_data = np.array([[0.9, 0.1, 0.7, 0.2],

                           [0.0, 0.9, 0.3, 0.8],

                           [0.2, 1.0, 0.0, 0.7],

                           [1.0, 0.0, 0.8, 0.1],

                           [0.3, 0.4, 0.3, 0.6]])

    expect_data = np.array([0, 1, 1, 0, 1])

    # Inicialización del modelo
    adaline = Adaline(input_length=4, learning_rate=.01)

    # Entrenamiento
    adaline.train(input_data, expect_data, 100)

    # Testear modelo
    test_dataset = list(zip(input_data, expect_data))
    shuffle(test_dataset)
    for ts_input, expected in test_dataset:
        output = adaline.predict(ts_input)
        expected = 'OK' if expected == output else 'FAIL'
        print(f'Input:{ts_input} Output: {output} = {expected}')

    # Mostrar error
    adaline.show_error()
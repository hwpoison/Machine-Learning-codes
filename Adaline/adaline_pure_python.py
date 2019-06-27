#!/usr/bin/env python
from __future__ import print_function
from random import shuffle, uniform
import random
import time

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

    def dot(self, a, b):
        # zigma
        return sum(x*z for x, z in zip(a, b))

    def forward(self, inputs):
        # propagación
        return self.dot(self.synapse_weights, inputs) + self.bias

    def predict(self, inputs):
        return 0 if self.forward(inputs) < 0.0 else 1

    def activation(self, inputs):
        # activación
        return self.predict(inputs)

    def train(self, input_data, expected_output, epochs=10):
        # Entrenamiento por gradiente descendente estocastica

        # Se inicializan los pesos
        self.synapse_weights = [random.uniform(
            0.001, 0.001) for i in range(0, self.input_length)]
        expected_output = [-1 if i == 0 else 1 for i in expected_output]
        time_init = time.time()
        # Comienzo de ciclos de aprendizaje
        for epoch in range(epochs):
            # Se elige un lote y se desordena
            batch = [i for i in range(len(expected_output)-1)]
            shuffle(batch)
            for element_index in batch:
                # Se calcula el peso postsinaptico
                output = self.forward(input_data[element_index])
                # Se calcular el error
                error = expected_output[element_index] - output
                # Se actualizan los pesos y el sesgo
                # en función a la gradiente descendente
                for w in range(0, self.input_length):
                    update_value = self.learning_rate * \
                        self.dot([input_data[element_index][w]], [error])
                    self.synapse_weights[w] += update_value
                    self.bias += self.learning_rate * error
            # Se calcula el error cuadratico medio / función de coste
            cost = [(e-i)**2 for e, i in zip(expected_output,
                                             list(map(self.forward, input_data)))]
            sum_cost = sum(cost) / 2.0
            self.all_errors.append(sum_cost)
        time_final = time.time() - time_init
        print(f"Trained in:{ time_final } seconds.\nMin Error: {sum_cost}")

    def show_error(self):
        print(f"Min error:{self.all_errors[-1:][0]}")


if __name__ == '__main__':
    # Dataset de prueba
    ts_inputs = [[0.9, 0.1, 0.7, 0.2],

                 [0.0, 0.9, 0.3, 0.8],

                 [0.2, 1.0, 0.0, 0.7],

                 [1.0, 0.0, 0.8, 0.1],

                 [0.3, 0.4, 0.3, 0.6]]

    ts_outputs = [0, 1, 1, 0, 1]
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

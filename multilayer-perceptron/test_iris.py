#!/usr/bin/env python
import numpy as np
from multilayer_perceptron import *

"""
Iris setosa:
    1-sepal length in cm
    2-sepal width in cm
    3-petal length in cm
    4-petal width in cm
    5-class ( Setosa, Vericolour, Iris Virginica)
download from: http://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data
"""


def generate_iris_dataset():
    # open and parser iris.data file
    iris_data = []
    with open('iris.data', 'r') as iris:
        for iris in iris.read().splitlines():
            if not iris:
                break
            data = [i if i[0].isalpha() else float(i) for i in iris.split(',')]
            if('Iris-setosa' in data):
                iris_data.append([data[:4], [.9, 0.0, 0.0]])
            elif('Iris-versicolor' in data):
                iris_data.append([data[:4], [0.0, .9, 0.0]])
            # For a no linear problem classification
            elif('Iris-virginica' in data):
                iris_data.append([data[:4], [0.0, 0.0, .9]])

    return iris_data


iris_dataset = generate_iris_dataset()
ts_input_iris = np.array([specs[0] for specs in iris_dataset])  # input data
ts_output_iris = np.array([[specs[1]
                            for specs in iris_dataset]])[0]  # expected output data

model = [
    NeuralLayer(4, 8),
    NeuralLayer(8, 3)
]
neural_network = NeuralNetwork(model)
neural_network.learn_rate = 0.001
neural_network.train(ts_input_iris, ts_output_iris, 5000)

print(neural_network.errors[-1])


def test_all():
    cout = 0
    ok = fails = 0
    for ts_input, expected in zip(ts_input_iris, ts_output_iris):
        output = neural_network.input(ts_input)
        iris_type = np.rint(output)
        iris_type_name = "Unknow"
        if iris_type[0]:
            iris_type_name = "Setosa"
        elif iris_type[1]:
            iris_type_name = "Versicolor"
        elif iris_type[2]:
            iris_type_name = "Virginica"
        validation = 1 if (np.rint(expected) == iris_type).all() else 0
        if validation:
            ok+=1
        else:
            fails+=1
    
        print(f'{cout})Input:{ts_input} Output: {output} = {iris_type_name} { validation }  ')
        cout += 1
    print(f"[ OK:{ ok } Fail:{ fails } ]")
    neural_network.show_error()
test_all()


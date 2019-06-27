#!/usr/bin/env python
import numpy as np
from adaline import Adaline

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
                iris_data.append([data[:4], 1])
            elif('Iris-versicolor' in data):
                iris_data.append([data[:4], 0])

            # For a no linear problem classification
            # elif('Iris-virginica' in data):
            #	iris_data.append([data[:4], 3])
    return iris_data


# instance adaline
adaline = Adaline(input_length = 4, learning_rate=0.01)
adaline.input_length = 4  # set input length

# prepare dataset
iris_dataset = generate_iris_dataset()
ts_input_iris = np.array([specs[0] for specs in iris_dataset])  # input data
ts_output_iris = np.array([specs[1]
                           for specs in iris_dataset])  # expected output data

# train
adaline.train(ts_input_iris, ts_output_iris, 10)

# test al dataset


def test_all():
    for ts_input, expected in zip(ts_input_iris, ts_output_iris):
        output = adaline.predict(ts_input)
        expected = 'OK' if expected == output else 'FAIL'
        iris_type_name = "Iris-setosa" if output == 1 else "Iris-versicolor"
        print(f'Input:{ts_input} Output: {output} = {iris_type_name}')

test_all()
# you can test a iris measurements
p_input = [5.9, 3.0, 4.2,0.0]
if(adaline.predict(p_input) == 1):
    print("Is a Iris setosa")
else:
    print("Is a Iris versicolor")
adaline.show_error()
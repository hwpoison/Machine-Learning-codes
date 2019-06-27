from perceptron import Perceptron
import numpy as np

# Problema de clasificación lineal
# dataset ficticio para clasificacion de maduración de tomates
# verdor| rojez | consistencia | suavidad


#entradas de muestras
ts_inputs = np.array([[0.9, 0.1, 0.7, 0.2],

                      [0.0, 0.9, 0.3, 0.8],

                      [0.2, 1.0, 0.0, 0.7],

                      [1.0, 0.0, 0.8, 0.1],

                      [0.3, 0.4, 0.3, 0.6],

                      [0.5, 0.6, 0.4, 0.4]])
# salidas esperadas
ts_outputs = np.array([0, 1, 1, 0, 1, 0]).T

# instanciar perceptron
perceptron = Perceptron()
print("Inicializando pesos...")
perceptron.synapse_weights = [0, 0, 0, 0]
print("Asignando tasa de aprendizaje y bias")
perceptron.learn_rate = 0.01
perceptron.bias = 1.0

# entrenamiento
print("Entrenando...")
perceptron.train(ts_inputs, ts_outputs)
print("Entrenamiento finalizado...")

def detect_tomate(salida):
    if(salida):
        return "Tomate maduro"
    else:
        return "Tomate Verde"
print("Inserte las propiedades de su tomate!:")
while True:
    try:
        verdor = float(input("Cantidad de verdor:"))
        rojez = float(input("Cantidad de rojez:"))
        consistencia = float(input("Consistencia:"))
        suavidad = float(input("Suavidad:"))
        salida = detect_tomate(perceptron.predict(
            [verdor, rojez, consistencia, suavidad]))
        print("Es un", salida)
    except ValueError as error:
        print(f"Por favor introduce un numero valido. [{error}]")

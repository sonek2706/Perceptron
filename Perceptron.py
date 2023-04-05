import matplotlib.pyplot as plt
import numpy as np
from math import sqrt
import random

from dataclasses import dataclass

@dataclass
class data_Node:

    n_columns: int
    value: str
    data: list

    def __init__ (self, data):
        self.n_columns = len(data) - 1
        self.type = data[-1].strip()      
        self.value = 1 if self.type == "Iris-versicolor" else 0
        self.data = list(map(float, data[:-1]))

    def __str__(self):
        return f"I'm {self.type}, value={self.value}"

def calculate_net(x: list, weight: list, bias : float):
    return np.dot(np.array(weight).T.tolist(), np.array(x).tolist()) - bias

def activation(x: data_Node, weight: list, bias : float):
    net = calculate_net(x.data, weight, bias)
    return 1 if net>=0 else 0, net

def perceptron(weight: int, bias: float, learning_rate:float, max_iterations:int)->list:
    for i in range(len(input)):
        x = input[i]
        y, net = activation(x, weight, bias)

        if (x.value != y):
            error = input[i].value - net
            for j in range(x.n_columns):
                weight[j] += learning_rate*error*x.data[j]

            bias -= learning_rate*error

    return weight


if __name__ == "__main__":
    # READING DATA
    input = []
    with open ("perceptron.data") as data:
        for line in data:
            tmp = data_Node(line.split(","))
            input.append(tmp)

    # READING DATA
    test = []
    with open("perceptron.test.data") as test_data:
        for line in test_data:
            tmp = data_Node(line.split(","))
            test.append(tmp)


    accuracy = []

    learning_rate = 0.01
    max_iterations = 100
    for i in range(1000):
        weight = [random.random() for i in range(len(input[0].data))]
        bias = random.random()
        
        #Calculate the weight for perceptron.data
        weight = perceptron(weight, bias, learning_rate, max_iterations)

        # TESTING
        counter = 0
        for x in test:
            y, net = activation(x, weight, bias)
            if (x.value == y):
                counter += 1
        
        accuracy.append(counter/len(test)*100)

    plt.plot([n for n in range(1,1000+1)], accuracy)
    plt.ylabel('Accuracy')
    plt.show()

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
        return f"I'm {self.type}"


def activation(x: data_Node, weight: list, bias : float):
    net = np.dot(weight, x.data)
    return 1 if net>=0 else 0, net


if __name__ == "__main__":
    # READING DATA
    input = []
    with open ("perceptron.data") as data:
        for line in data:
            tmp = data_Node(line.split(","))
            input.append(tmp)

    # LEARNING
    w = [random.random() for i in range(len(input[0].data))]
    bias = random.random()
    learning_rate = 0.01
    max_iterations = 100

    for i in range(len(input)):
        x = input[i]
        y, net = activation(x, w, bias)

        if (x.value == y):
            pass
        else:        
            error = input[i].value - net
            for j in range(x.n_columns):
                w[j] += learning_rate*error*x.data[j]

            bias -= learning_rate*error

    # READING DATA
    test = []
    with open("perceptron.test.data") as test_data:
        for line in test_data:
            tmp = data_Node(line.split(","))
            test.append(tmp)

    # TESTING
    counter = 0
    for x in test:
        y, net = activation(x, w, bias)
        if (x.value == y):
            counter += 1
    
    print(counter/len(test))

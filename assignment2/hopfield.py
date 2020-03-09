#!/usr/bin/python3

import numpy as np
import pandas as pd
from matplotlib import colors
from matplotlib import pyplot as plt

def activation_function(f, net):
    if net > 0:
        return 1
    elif net == 0:
        return f
    else:
        return -1

def plotting(vector, shape):
    cmap = colors.ListedColormap(['black', 'white'])
    bounds = [-1.5, 0, 1.5]
    norm = colors.BoundaryNorm(bounds, cmap.N)

    _ , ax = plt.subplots()
    ax.imshow(np.array(vector).reshape(shape), cmap=cmap, norm=norm)

    ax.grid(linewidth=0.2)
    ax.set_xticks(np.arange(-0.5, 9, 1))
    ax.set_yticks(np.arange(-0.5, 10, 1))

    plt.xlabel("Width")
    plt.ylabel("Height")
    plt.title("Alphabet")
    plt.show()

    return

class Hopfield():
    def __init__(self, nodes):
        self.nodes = nodes
        self.patterns = 0
        self.weights = np.zeros((nodes, nodes))

    def __str__(self):
        desc = "nodes: "+str(self.nodes)+", patterns: "+str(self.patterns)
        return "Hopfield("+desc+")\n"

    def __energy(self, vector):
        e = 0
        for i in range(self.nodes):
            for j in range(self.nodes):
                e += self.weights[i][j]*vector[i]*vector[j]
        return -1*e/2

    def stabilize(self, data):
        self.patterns = len(data)
        for i in range(self.patterns):
            self.weights += np.dot(data[i].reshape(-1, 1), data[i].reshape(1, -1))

        for i in range(self.patterns):
            for j in range(self.patterns):
                if i == j:
                    self.weights[i][j] = 0
                else:
                    self.weights[i][j] += self.weights[i][j]/self.patterns

        return

    def predict(self, y, update="async"):
        energy_pre = -1
        energy_cur = self.__energy(y)
        tmp = y.copy()
        converges = 0

        while energy_pre != energy_cur:
            if update == 'async':
                for i in range(len(y)):
                    tmp2 = 0
                    for j in range(len(y)):
                        tmp2 += self.weights[i][j] * tmp[j]
                    tmp[i] = activation_function(tmp[i], tmp2)
                y = tmp.copy()
            elif update == 'sync':
                tmp = y.copy()
                for i in range(len(y)):
                    tmp2 = 0
                    for j in range(len(y)):
                        tmp2 += self.weights[i][j] * y[j]
                    tmp[i] = activation_function(y[i], tmp2)
                y = tmp.copy()

            energy_pre = energy_cur
            energy_cur = self.__energy(y)
            converges += 1
        print("Converged in {} iterations".format(converges))
        return y

if __name__ == '__main__':
    # fname = input('Enter the name of the file: ')
    fname = 'training_data.csv'

    train = pd.read_csv(fname)

    y = train['label']
    train.drop(['label'], axis=1, inplace=True)
    x = train.values

    network = Hopfield(100)
    print(network)
    network.stabilize(x)
    print(network)

    test = pd.read_csv('distorted.csv')
    test.drop(['label'], axis=1, inplace=True)
    output = network.predict(test.loc[0, :].values)
    plotting(test.loc[0, :].values, (10, 10))
    plotting(output, (10, 10))
    print(output)
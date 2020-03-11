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

def plotting(actual, predict, shape):
    cmap = colors.ListedColormap(['black', 'white'])
    bounds = [-1.5, 0, 1.5]
    norm = colors.BoundaryNorm(bounds, cmap.N)

    _ , ax = plt.subplots(nrows=1, ncols=2, sharex=True, sharey=True)
    ax[0].imshow(np.array(predict).reshape(shape), cmap=cmap, norm=norm)
    ax[1].imshow(np.array(actual).reshape(shape), cmap=cmap, norm=norm)

    for i in range(2):
        ax[i].grid(linewidth=0.2)
        ax[i].set_xticks(np.arange(-0.5, 9, 1))
        ax[i].set_yticks(np.arange(-0.5, 10, 1))
        ax[i].set(xlabel='Width', ylabel='Height')
    ax[0].set_title("Corrupted Letter")
    ax[1].set_title("Converged Letter")

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

    def __async(self, vector_in):
        for i in range(len(y)):
            tmp = 0
            for j in range(len(y)):
                tmp += self.weights[i][j] * vector_in[j]
            vector_in[i] = activation_function(vector_in[i], tmp)

        return vector_in.copy()

    def __sync(self, vector_in):
        tmp = vector_in.copy()
        for i in range(len(y)):
            tmp2 = 0
            for j in range(len(y)):
                tmp2 += self.weights[i][j] * vector_in[j]
            tmp[i] = activation_function(vector_in[i], tmp2)

        return tmp.copy()

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
                y = self.__async(tmp)
            elif update == 'sync':
                y = self.__async(y)

            energy_pre = energy_cur
            energy_cur = self.__energy(y)
            converges += 1
        print("Converged in {} iterations".format(converges))
        return y

if __name__ == '__main__':
    fname = input('Enter distorted file-name: ')

    train = pd.read_csv('training_data.csv')

    y = train['label']
    train.drop(['label'], axis=1, inplace=True)
    x = train.values

    network = Hopfield(100)
    print(network)
    network.stabilize(x)
    print(network)

    test = pd.read_csv(fname)
    test.drop(['label'], axis=1, inplace=True)
    output = network.predict(test.values[0])
    plotting(output, test.values[0], (10, 10))
    print("Converged Letter:\n", output.reshape(10, 10))
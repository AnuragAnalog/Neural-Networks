#!/usr/bin/python3

import numpy as np
import pandas as pd
from matplotlib import colors
from matplotlib import pyplot as plt

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
    def __init__(self, shape):
        self.shape = shape
        self.nodes = np.prod(self.shape)
        self.patterns = 0
        self.weights = np.zeros((self.nodes, self.nodes))

    def __str__(self):
        desc = "nodes: "+str(self.nodes)+", patterns: "+str(self.patterns)
        return "Hopfield("+desc+")\n"

    def __energy(self, vector):
        # calculate the energy of the given vector
        e = 0
        for i in range(self.nodes):
            for j in range(self.nodes):
                e += self.weights[i][j]*vector[i]*vector[j]
        return -1*e/2

    def __async(self, vector_in):
        # Asynchronously update the weights of the vector
        for i in range(len(y)):
            tmp = 0
            for j in range(len(y)):
                tmp += self.weights[i][j] * vector_in[j]
            vector_in[i] = self.__activation_function(vector_in[i], tmp)

        return vector_in.copy()

    def __sync(self, vector_in):
        # Synchronously update the weights of the vector
        tmp = vector_in.copy()
        for i in range(len(y)):
            tmp2 = 0
            for j in range(len(y)):
                tmp2 += self.weights[i][j] * vector_in[j]
            tmp[i] = self.__activation_function(vector_in[i], tmp2)

        return tmp.copy()

    def __activation_function(self, f, net):
        if net > 0:
            return 1
        elif net == 0:
            return f
        else:
            return -1

    def stabilize(self, data):
        """
        data : ndarray, list
            Training data
        """
        self.patterns = len(data)
        for i in range(self.patterns):
            self.weights += np.dot(data[i].reshape(-1, 1), data[i].reshape(1, -1))

        # Constructing the weight matrix
        for i in range(self.patterns):
            for j in range(self.patterns):
                if i == j:
                    self.weights[i][j] = 0
                else:
                    self.weights[i][j] += self.weights[i][j]/self.patterns

        return

    def predict(self, y, update="async"):
        """
        y : ndarray, list
            Input vecto,r
        update : str, optional
            updating the input vector synchronsly or asynchronously
            valid arguments are "sync" and "async"
        """
        tmp = np.empty(y.shape)
        converges = 0

        # Run the loop until the energy of previous and current are equal
        while all(np.array(tmp) == np.array(y)) is False:
            tmp = y.copy()
            if update == 'async':
                y = self.__async(tmp)
            elif update == 'sync':
                y = self.__async(y)
            print(tmp, y)

            converges += 1

        print("Converged in {} iterations with energy {}".format(converges, self.__energy(y)))
        return y

if __name__ == '__main__': # This code is for testing
    ftrain = input("Enter the training filename: ")
    ftest = input('Enter distorted file-name: ')
    ftrain = "training_data.csv"
    ftest = "distorted.csv"

    train = pd.read_csv(ftrain)

    y = train['label']
    train.drop(['label'], axis=1, inplace=True)
    x = train.values

    network = Hopfield((10, 10))
    print(network)
    network.stabilize(x)
    print(network)

    test = pd.read_csv(ftest)
    test.drop(['label'], axis=1, inplace=True)
    output = network.predict(test.values[1], update="sync")
    plotting(output, test.values[1], (10, 10))
    print("Converged Letter:\n", output.reshape(10, 10))
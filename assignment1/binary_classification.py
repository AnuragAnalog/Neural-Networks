#!/usr/bin/python3

import sys
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from metrics import MSE
from encoding import OneHotEncoder
from activation_function import sigmoid

class Classifier():
    def __init__(self, features, labels, hidden_layer=[], learning_rate=0.01):
        self.epochs = 10
        self.lr = learning_rate
        self.input_layer = features + 1
        self.hidden_layers = hidden_layer
        self.output_layer = labels
        self.layers = len(hidden_layer) + 2
        self.nodes = [self.input_layer]
        for n in self.hidden_layers:
            self.nodes = self.nodes + [n+1]
        self.nodes = self.nodes + [self.output_layer]
        self.data_size = 0
        self.weights = list()
        self.inputs_at_layers = list()
        self.activations = list()
        self.training_loss = list()

    def __str__(self):
        about_hyperp = "Epochs: "+str(self.epochs)+", Learning Rate: "+str(self.lr)
        about_layers = ", Layers: "+str(self.layers)
        return "Classifier("+about_hyperp+about_layers+")\n"

    def __initialization(self, row, col, variant='he', bias=False):
        """
        Parameters
        ----------
        row : int
            Number of rows for the weight matrix
        col : int
            Number of columns for the weight matrix
        variant : string, optinal
            Type of weight initialization
        bias : bool, optional
            Initialize bias with random weights if True

        Returns
        -------
        w : ndarray
            An Randomly initialized array object
        """
        if variant == 'he':
            w = np.random.randn(row, col) * np.sqrt(2 / row)
        elif variant == 'xavier':
            w = np.random.randn(row, col) * np.sqrt(1 / row)
        elif variant == 'random':
            w = np.random.randn(row, col)
        elif variant == 'zero':
            w = np.zeros((row, col))

        if bias is False:
            w[-1, :] = [0] * col

        return w
    
    def model(self, x, y, epochs=100):
        """
        Parameters
        ----------
        x : ndarray
            Features of the training data
        y : ndarray
            Labels of the training data
        epochs : int, optional
            Number of epochs

        Returns
        -------
        None : NoneType
        """
        self.epochs = epochs
        self.data_size = len(x)
        for i in range(self.layers-2):
            self.weights.append(self.__initialization(self.nodes[i], self.nodes[i+1]-1))
        self.weights.append(self.__initialization(self.nodes[i+1], self.output_layer))

        for e in range(self.epochs):
            predict = list()
            for i in range(self.data_size):
                tmp = [x[i]]
                # Forward pass
                for n in range(self.layers-1):
                    self.activations.append(list(tmp[0]) + [-1])
                    tmp = sigmoid(np.dot(np.array(list(tmp[0]) + [-1]).reshape(1, self.nodes[n]), self.weights[n]))
                output = tmp[0]
                predict.append(output)

                # Backward pass
                del_weights0 = [[0 for u in range(self.nodes[-2])] for v in range(self.nodes[-1])]
                for u in range(self.nodes[-1]):
                    for v in range(self.nodes[-2]):
                        del_weights0[u][v] += self.lr * (y[i][u] - output[u]) * output[u] * (1 - output[u]) * self.activations[-1][v]
                del_weights1 = [[0 for u in range(self.nodes[-3])] for v in range(self.nodes[-2]-1)]
                for q in range(self.nodes[-3]):
                    for w in range(self.nodes[-2]-1):
                        for z in range(self.nodes[-1]):
                            del_weights1 += self.lr * (y[i][z] - output[z]) * output[u] * (1 - output[u]) * self.weights[1][w][z] * self.activations[-1][w] * (1-self.activations[-1][w])*self.activations[-2][q]
            del_weights0 = np.array(del_weights0).reshape(self.nodes[-2], self.nodes[-1])
            for u in range(self.nodes[-2]):
                for v in range(self.nodes[-1]):
                    self.weights[-1][u][v] += del_weights0[u][v]
            for u in range(self.nodes[-3]):
                for v in range(self.nodes[-2]-1):
                    self.weights[-2][u][v] += del_weights1[u][v]
            predict = np.array(predict).reshape(self.data_size, self.output_layer)
            self.training_loss.append(MSE(y, predict))
            print(f"Epochs: {e+1}/{self.epochs}.. Training Loss: {self.training_loss[e]}..")

    def predict(self, y_new):
        if len(y_new) != self.input_layer:
            print("Dimension mismatch")
            sys.exit()

        predicted = np.array(y_new.append(1)).reshape(1, self.input_layer)
        for i in range(self.layers):
            predicted = np.dot(predicted, self.weights[i])

        return predicted

if __name__ == '__main__':
    fname = 'mnist_train.csv'
    data = pd.read_csv(fname)

    ### Preproscessing
    encode = OneHotEncoder()

    encode.fit(data.label)
    labels = encode.transform(data.label)

    data.drop(['label'], axis=1, inplace=True)
    x = data.values
    y = labels

    ### Running the model
    network = Classifier(2, 2, [3], learning_rate=0.03)
    print(network)
    network.model(x, y, 1000)
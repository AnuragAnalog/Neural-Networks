#!/usr/bin/python3

import sys
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

# Custom modules
from metrics import MSE
from encoding import OneHotEncoder, categorize
from splitting import train_test_split
from activation_function import sigmoid

class Perceptron():
    """
        An implementation of a single layer perceptron, which takes two features(Mass, Speed) as input and predicts the class(Bomber or Fighter) of the plane
    """

    def __init__(self, in_layer, out_layer, learning_rate=0.01):
        self.input_layer = in_layer + 1
        self.output_layer = out_layer
        self.lr = learning_rate
        self.epochs = 100
        self.weights = list()
        self.del_weights = list()
        self.training_loss = list()

    def __str__(self):
        about_hyperparameters = "Epochs: "+str(self.epochs)+" Learning Rate: "+str(self.lr)
        return "Perceptron("+about_hyperparameters+")"

    def __initialization(self, row, col, variant='he', bias=False):
        # Initializing the weight matrix with random values.

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

    def __add_bais(self, vector):
        # Add a bais Transforms a vector into the required shape

        vector_in = list(vector) + [-1.0]
        return vector_in

    def forward(self, x):
        output = sigmoid(np.dot(self.weights.T, x))

        return output

    def backward(self, in_value, output, actual):
        self.del_weights = np.zeros(self.weights.shape)

        for i in range(self.input_layer):
            for j in range(self.output_layer):
                if self.learning == 'gradient':
                    self.del_weights[i][j] += self.lr * (actual[j] - output[j]) * output[j] * (1 - output[j]) * in_value[i]
                elif self.learning == 'perceptron':
                    self.del_weights[i][j] += self.lr * (actual[j] - output[j]) * in_value[i]
                else:
                    print("Not a valid learning algorithm.")
                    sys.exit()

        return

    def update_weights(self):
        for i in range(self.input_layer):
            for j in range(self.output_layer):
                self.weights[i][j] += self.del_weights[i][j]

        return

    def model(self, x, y, epochs=100, learning='gradient'):
        """
        Find the weights

        Parameters
        ----------
        x : ndarray, list
            Features of the training data
        y : ndarray, list
            Labels of the training data
        epochs : int, optional
            Number of epochs to run
        learning : str, optional
            Type of the learning

        Return
        ------
        None : NoneType
        """

        self.epochs = epochs
        self.learning = learning
        self.data_length = len(x)
        self.weights = self.__initialization(self.input_layer, self.output_layer, variant='random')
        print(self.weights, self.weights.shape)

        for e in range(self.epochs):
            predict = list()
            for i in range(self.data_length):
                x_tmp = self.__add_bais(x[i])
                output = self.forward(x_tmp)
                self.backward(x_tmp, output, y[i])

                predict.append(output.tolist())
            self.update_weights()

            self.training_loss.append(MSE(y, predict))
            print(f"Epochs: {e+1}/{epochs}.. Training Loss: {self.training_loss[e]: 3f}..\nWeights: {self.weights.tolist()}\n")

        return

    def plotting(self):
        """
        Parameters
        ----------
        train : ndarray, list
            Training data
        test : ndarray, list
            Testing data

        Return
        ------
        None : NoneType
        """

        plt.figure()
        plt.grid(True)
        plt.title("Training Loss")
        plt.xlabel("Epochs")
        plt.ylabel("Loss(MSE)")
        plt.plot(range(self.epochs), self.training_loss)
        plt.show()

        return

    def predict(self, y_new):
        """
        Takes an observation as input and classifies it

        Parameters
        ----------
        y_new : ndarray, list
            An observation

        Returns
        -------
        predicted : ndarray
            The output vector
        """

        y_tmp = self.__add_bais(y_new)
        predicted = self.forward(y_tmp)
        print(predicted)

        return predicted

if __name__ == '__main__':
    # fname = input("Enter the filename: ")
    fname = 'dataset.csv'
    data = pd.read_csv(fname)

    ## Preprocessing of data
    train, test = train_test_split(data, train_ratio=0.8)

    # cat = OneHotEncoder()
    cat = categorize()
    cat.fit(train[:, 2])
    train_y = cat.transform(train[:, 2])
    train_x = train[:, [0, 1]]

    ## Run the model
    epochs = int(input("Enter the number of Epochs: "))
    learning_rate = float(input("Enter the Learning rate: "))

    net = Perceptron(2, 2, learning_rate=learning_rate)
    print(net)
    net.model(train_x, train_y, epochs=epochs)
    print(net)
    net.plotting()

    ## Prediction
    net.predict([0.50, 0.45])

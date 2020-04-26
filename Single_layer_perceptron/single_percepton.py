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

np.random.seed(4)

class Perceptron():
    """
        An implementation of a single layer perceptron.
    """

    def __init__(self, in_layer, out_layer, learning_rate=0.01, variant='random'):
        self.model_ran = False
        self.input = in_layer
        self.output = out_layer
        self.lr = learning_rate
        self.epochs = 10
        self.zero_grad()
        self.__initialization(self.input, self.output, variant=variant)

    def __str__(self):
        about_hyperparameters = "Epochs: "+str(self.epochs)+" Learning Rate: "+str(self.lr)
        return "Perceptron("+about_hyperparameters+")"

    def __initialization(self, row, col, variant='he', bias=False):
        # Initializing the weight matrix with random values.

        if variant == 'he':
            self.W = np.random.randn(row, col) * np.sqrt(2 / row)
        elif variant == 'xavier':
            self.W = np.random.randn(row, col) * np.sqrt(1 / row)
        elif variant == 'random':
            self.W = np.random.randn(row, col)
        elif variant == 'zero':
            self.W = np.zeros((row, col))

        if bias:
            self.b = np.random.randn(self.output).reshape((self.output, -1))
        else:
            self.b = np.zeros(self.output).reshape((self.output, -1))

        return

    def __reshape_input(self, vector):
        # Transforms a vector into the required shape

        if isinstance(vector, list):
            vector = np.array(vector)

        vector_in = vector.reshape((self.input, -1))
        return vector_in

    def __reshape_output(self, vector):
        # Transforms a vector into the required shape

        if isinstance(vector, list):
            vector = np.array(vector)

        vector_in = vector.reshape((self.output, -1))
        return vector_in

    def forward(self, x):
        """
        Forward pass in the Neural network

        Parameters
        ----------
        x : ndarray, list
            An input training example

        Return
        ------
        output : ndarray
            An output vector
        """

        output = sigmoid(np.dot(self.W.T, x) + self.b)

        return output

    def backward(self, in_value, output, actual):
        """
        One backward propagation

        Parameters
        ----------
        in_value : ndarray, list
            An input training example
        output : ndarray
            An output vector obtained from forward pass
        actual : ndarray, list
            The Actual label for the training example

        Return
        ------
        None : NoneType
        """

        self.dW += self.lr * np.dot(((actual - output) * output * (1 - output)), in_value.reshape((-1, self.input))).T
        self.db += self.lr * (actual - output) * output * (1 - output)

        if self.dW.dtype == 'O':
            self.dW = self.dW.astype("float64")

        return

    def update_weights(self):
        """
        Update the weights
        """

        self.W += self.dW
        self.b += self.db

        return

    def model(self, x, y, epochs=100, debug=False, debug_verbose=False):
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
        self.model_ran = True
        self.data_length = len(x)
        self.training_loss = list()

        for e in range(self.epochs):
            predict = list()
            self.zero_grad()
            for i in range(self.data_length):
                x_tmp = self.__reshape_input(x[i])
                output = self.forward(x_tmp)
                y_tmp = self.__reshape_output(y[i])
                self.backward(x_tmp, output, y_tmp)

                predict.append(output.tolist())
            self.update_weights()

            if debug:
                self.debug(more_verbose=debug_verbose)

            self.training_loss.append(MSE(y, predict))
            print(f"Epochs: {e+1}/{epochs} Training Loss: {self.training_loss[e]}..")

        return

    def zero_grad(self):
        self.dW = 0
        self.db = 0

        return

    def debug(self, more_verbose=False):
        print("Weights are: {}".format(self.W))
        print("Biases are: {}".format(self.b))

        if more_verbose:
            print("dW is: {}".format(self.dW))
            print("db is: {}".format(self.db))

    def get_weights(self):
        return self.W

    def get_biases(self):
        return self.b

    def summary(self):
        if self.model_ran is False:
            print("Run the model before viewing summary statistics")
            return
        
        print("=============================")
        print("\tSummary\t")
        print("=============================")
        print("Number of features are {}\t".format(self.input))
        print("Number of targets are {}\t".format(self.output))
        print("Model ran for {} epochs\t".format(self.epochs))
        print("Trained on {} examples\t".format(self.data_length))
        print("With a learning rate of\t{}".format(self.lr))
        print("Dimensions of weights {}\t".format(self.W.shape))
        print("Dimensions of biases {}\t".format(self.b.shape))
        print("=============================")

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

        y_tmp = self.__reshape_input(y_new)
        predicted = self.forward(y_tmp)
        print(predicted)

        return predicted

if __name__ == '__main__':
    # fname = input("Enter the filename: ")
    fname = 'dataset.csv'
    data = pd.read_csv(fname)

    cat = OneHotEncoder()
    # cat = categorize()
    cat.fit(data['Class'].values)
    y = cat.transform(data['Class'].values)
    x = data[['Mass', 'Speed']].values

    ## Run the model
    epochs = 10000
    learning_rate = 0.1
    # epochs = int(input("Enter the number of Epochs: "))
    # learning_rate = float(input("Enter the Learning rate: "))

    net = Perceptron(2, 2, learning_rate=learning_rate)
    print(net)
    net.model(x, y, epochs=epochs)
    print(net)
    net.summary()
    net.plotting()

    ## Prediction
    net.predict(x[1])

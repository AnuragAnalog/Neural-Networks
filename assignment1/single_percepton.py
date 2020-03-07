#!/usr/bin/python3

"""
@Anurag.peddi, 17MCME13
"""

import sys
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

# Custom modules
from metrics import MSE
from encoding import categorize
from splitting import train_test_split
from activation_function import sigmoid

class Perceptron():
    """
        An implementation of a single layer perceptron, which takes two features(Mass, Speed) as input and predicts the class(Bomber or Fighter) of the plane
    """
    def __init__(self, in_layer, learning_rate=0.01):
        self.input_layer = in_layer + 1
        self.lr = learning_rate
        self.epochs = 100
        self.weights = list()
        self.slope = 0
        self.intercept = 0
        self.training_loss = list()

    def __str__(self):
        about_hyperparameters = "Epochs: "+str(self.epochs)+" Learning Rate: "+str(self.lr)
        about_coef = "(Slope, Intercept) -> ("+str((self.slope, self.intercept))+")"
        return "Perceptron("+about_hyperparameters+"\n"+about_coef+")"
    
    def model(self, x, y, epochs=100, learning='gradient'):
        """
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
        self.weights = np.random.randn(self.input_layer, 1)

        for e in range(self.epochs):
            del_weights = np.zeros(self.weights.shape)
            predict = list()
            for i in range(len(x)):
                x_tmp = list(x[i]) + [-1]
                output = sigmoid(np.dot(np.array(x_tmp).reshape(1, self.input_layer), self.weights))
                predict.append(output[0][0])
                for j in range(self.input_layer):
                    if self.learning == 'gradient':
                        del_weights[j] += self.lr * (y[i] - output[0]) * output[0] * (1 - output[0]) * x_tmp[j]
                    elif self.learning == 'perceptron':
                        del_weights[j] += self.lr * (y[i] - output[0]) * x_tmp[j]
                    else:
                        print("Not a valid learning algorithm.")
                        sys.exit()
            for j in range(self.input_layer):
                self.weights[j] += del_weights[j]

            self.training_loss.append(MSE(y, predict))
            print(f"Epochs: {e+1}/{epochs}.. Training Loss: {self.training_loss[e]: 3f}..\nWeights: {self.weights.tolist()}\n")

        self.slope = (-1 * self.weights[0]/self.weights[1])[0]
        self.intercept = (self.weights[2]/self.weights[1])[0]
        return

    def plotting(self, train, test):
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
        is_bomber_train = (train[:, 2] == 'Bomber')
        is_fighter_train = (train[:, 2] == 'Fighter')

        is_bomber_test = (test[:, 2] == 'Bomber')
        is_fighter_test = (test[:, 2] == 'Fighter')

        plt.grid(True)
        plt.title("Bomber or Fighter")
        plt.xlabel("Mass of the Plane")
        plt.ylabel("Speed of the Plane")

        plt.scatter(x=train[:, 0][is_bomber_train], y=train[:, 1][is_bomber_train], c='red', alpha=0.5, label="Bomber(train)")
        plt.scatter(x=train[:, 0][is_fighter_train], y=train[:, 1][is_fighter_train], c='blue', alpha=0.5, label="Fighter(train)")
        plt.scatter(x=test[:, 0][is_bomber_test], y=test[:, 1][is_bomber_test], c='red', label="Bomber(test)")
        plt.scatter(x=test[:, 0][is_fighter_test], y=test[:, 1][is_fighter_test], c='blue', label="Fighter(test)")
        plt.legend()
        axes = plt.gca()
        x_vals = np.array(axes.get_xlim())
        y_vals = self.intercept + self.slope * x_vals
        plt.plot(x_vals, y_vals, '-')

        plt.figure()
        plt.grid(True)
        plt.title("Training Loss")
        plt.xlabel("Epochs")
        plt.ylabel("Loss(MSE)")
        plt.plot(range(self.epochs), self.training_loss)
        plt.show()

        return

    def predict(self, y_new):
        y_tmp = y_new + [-1]
        predicted = sigmoid(np.dot(np.array(y_tmp).reshape(1, self.input_layer), self.weights))
        if predicted > 0.5:
            print("Fighter Plane")
            print("Confidence: ", predicted)
        else:
            print("Bomber Plane")
            print("Confidence: ", 1 - predicted)

        return predicted

if __name__ == '__main__':
    fname = input("Enter the filename: ")
    data = pd.read_csv(fname)

    ## Preprocessing of data
    train, test = train_test_split(data, train_ratio=0.8)

    cat = categorize()
    cat.fit(train[:, 2])
    train_y = cat.transform(train[:, 2])
    train_x = train[:, [0, 1]]

    ## Run the model
    epochs = int(input("Enter the number of Epochs: "))
    learning_rate = float(input("Enter the Learning rate: "))

    net = Perceptron(2, learning_rate=learning_rate)
    print(net)
    net.model(train_x, train_y, epochs=epochs)
    print(net)
    net.plotting(train, test)

    ## Prediction
    net.predict([0.50, 0.45])

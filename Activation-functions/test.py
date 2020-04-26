#!/usr/bin/python3

import numpy as np
import matplotlib
from matplotlib import pyplot as plt

def function():
    x_range = np.arange(-3, 3, 0.001)
    y_range = np.array([x if x >= 0 else 0 for x in x_range])
    y_derv = np.array([1 if x >= 0 else 0 for x in x_range])

    plt.grid(True)
    plt.plot(x_range, y_range, label='ReLU')
    plt.plot(x_range, y_derv, label='It\'s derivative')
    plt.axhline(0, color='black', linewidth=0.85)
    plt.axvline(0, color='black', linewidth=0.85)
    plt.legend()
    plt.show()

if __name__ == '__main__':
    function()
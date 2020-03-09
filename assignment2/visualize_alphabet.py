#!/usr/bin/python3

import numpy as np
import pandas as pd
from matplotlib import colors
from matplotlib import pyplot as plt

def draw_alphabet(filename):
    pixels = pd.read_csv(filename, header=None)

    cmap = colors.ListedColormap(['black', 'white'])
    bounds = [0, 0.5, 1.5]
    norm = colors.BoundaryNorm(bounds, cmap.N)

    _ , ax = plt.subplots()
    ax.imshow(pixels, cmap=cmap, norm=norm)

    ax.grid(linewidth=0.2)
    ax.set_xticks(np.arange(-0.5, 9, 1))
    ax.set_yticks(np.arange(-0.5, 10, 1))

    plt.xlabel("Width")
    plt.ylabel("Height")
    plt.title("Alphabet")
    plt.show()

if __name__ == '__main__':
    # fname = input("Enter the file-name: ")
    fname = './dataset_m/c_alpha.csv'

    draw_alphabet(fname)
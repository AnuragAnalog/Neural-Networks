#!/usr/bin/python3

import pandas as pd
from matplotlib import pyplot as plt

def plot_data(fname: str) -> None:
    data = pd.read_csv(fname)

    is_bomber = (data.Class == "Bomber")
    is_fighter = (data.Class == "Fighter")

    plt.grid(True)
    plt.title("Bomber or Fighter")
    plt.xlabel("Mass of the Plane")
    plt.ylabel("Speed of the Plane")

    plt.scatter(x=data.Mass[is_bomber], y=data.Speed[is_bomber], alpha=0.7, label="Bomber")
    plt.scatter(x=data.Mass[is_fighter], y=data.Speed[is_fighter], alpha=0.7, label="Fighter")
    plt.legend()
    plt.show()

    return

if __name__ == '__main__':
    fname = input("Enter the filename: ")
    # fname = 'dataset.csv'
    plot_data(fname)
#!/usr/bin/python3

import numpy as np
import pandas as pd

def distort(fin_name, fout_name, samples=40, percent=20):
    data = pd.read_csv(fin_name)
    cols = len(data.columns) - 1

    labels = list(data['label'].values)
    rand_labels = np.random.choice(labels, size=samples)

    col_name = "label "
    for i in range(1, cols+1):
        col_name += "p"+str(i)+" "
    col_name = col_name[:-1]
    col_name = col_name.split(" ")

    values = list()

    for i in range(samples):
        indices = np.random.choice(range(cols), size=percent, replace=False)
        vector = data[data['label'] == rand_labels[i]].values[0].copy()
        for ind in indices:
            if vector[ind+1] == 1:
                vector[ind+1] = -1
            elif vector[ind+1] == -1:
                vector[ind+1] == 1
        values.append(vector)

    data = pd.DataFrame(index=range(samples), columns=col_name, data=values)
    data.to_csv(fout_name, index=False)

    return

if __name__ == '__main__':
    fin = input("Enter the training file: ")
    fout = input("Enter the distort file: ")

    distort(fin, fout)
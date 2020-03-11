#!/usr/bin/python3

import numpy as np
import pandas as pd

def generate_data(rows, cols, labels, fname, cat=[-1, 1]):
    col_name = "label "
    for i in range(1, cols+1):
        col_name += "p"+str(i)+" "
    col_name = col_name[:-1]
    col_name = col_name.split(" ")

    values = list()

    for i in range(rows):
        tmp_col = list()
        for _ in range(cols):
            tmp_col.append(np.random.choice(cat))
        values.append([labels[i]] + tmp_col)

    data = pd.DataFrame(index=range(rows), columns=col_name, data=values)
    data.to_csv(fname, index=False)

    return

if __name__ == '__main__':
    # fname = input("Enter the filename: ")
    fname = 'simulated_data.csv'

    generate_data(8, 100, ['a', 'b', 'a', 'a', 'b', 'c', 'c', 'c'], fname)
#!/usr/bin/python3

import numpy as np
import pandas as pd

def flatten(fout, cols):
    col_name = "label "
    for i in range(1, cols+1):
        col_name += "p"+str(i)+" "
    col_name = col_name[:-1]
    col_name = col_name.split(" ")

    values = list()
    for i in range(97, 123):
        data = pd.read_csv(chr(i)+"_alpha.csv", header=None)
        tmp = [chr(i)] + list(data.values.flatten())
        values.append(tmp)

    df = pd.DataFrame(data=values, columns=col_name, index=range(26))
    df.to_csv(fout, index=False)

    return

if __name__ == '__main__':
    fout = input("Enter the Output filename: ")

    flatten(fout, 100)
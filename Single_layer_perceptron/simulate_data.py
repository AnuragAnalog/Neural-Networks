#!/usr/bin/python3

import numpy as np
import pandas as pd

def simulate(data, n):
    is_bomber = (data.Class == 'Bomber')
    is_fighter = (data.Class == 'Fighter')

    min_bm, max_bm = min(data.Mass[is_bomber]), max(data.Mass[is_bomber])
    min_fm, max_fm = min(data.Mass[is_fighter]), max(data.Mass[is_fighter])

    min_bs, max_bs = min(data.Speed[is_bomber]), max(data.Speed[is_bomber])
    min_fs, max_fs = min(data.Speed[is_fighter]), max(data.Speed[is_fighter])

    values = list()
    for _ in range(int(n/2)):
        bomber_x = np.random.rand() * (max_bm - min_bm) + min_bm
        bomber_y = np.random.rand() * (max_bs - min_bs) + min_bs

        fighter_x = np.random.rand() * (max_fm - min_fm) + min_fm
        fighter_y = np.random.rand() * (max_fs - min_fs) + min_fs

        values.append([bomber_x, bomber_y, 'Bomber'])
        values.append([fighter_x, fighter_y, 'Fighter'])
    
    return np.array(values)

if __name__ == '__main__':
    fin_name = input("Enter the name of dataset: ")
    fout_name = input("Enter the name of simulated dataset: ")
    n = int(input("Enter number of data points: "))

    data = pd.read_csv(fin_name)
    values = simulate(data, n)

    df = pd.DataFrame(data=values, columns=['Mass', 'Speed', 'Class'])
    df.to_csv(fout_name, index=False)
#!/usr/bin/python3

import numpy as np

def train_test_split(data, train_ratio=0.8):
    data_size = len(data)
    train_samples = int(data_size * train_ratio)
    test_samples = data_size - train_samples
    train, test = [], []
    indices = list()
    for i in range(test_samples):
        index = np.random.randint(data_size)
        indices.append(index)
        test.append(data.loc[index].values)
    for i in range(data_size):
        if i not in indices:
            train.append(data.loc[i].values)

    return np.array(train), np.array(test)
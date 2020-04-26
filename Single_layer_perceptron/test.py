#!/usr/bin/python3

import numpy as np
import pandas as pd

from encoding import OneHotEncoder, categorize

np.random.seed(4)

data = pd.read_csv('dataset.csv')
x = data[['Mass', 'Speed']].values
cat = OneHotEncoder()
# cat = categorize()
cat.fit(data['Class'].values)
y = cat.transform(data['Class'].values)

def model(x, y, lr, i, o): 
    w = np.random.randn(i, o)
    dw = np.zeros_like(w)
    b = np.zeros(o).reshape((o, -1))
    db = np.zeros_like(b)
    for e in range(10000): 
        for ii in range(10): 
            xt = x[ii].reshape((i, -1))
            tmp = np.dot(w.T, xt) + b 
            tmp = tmp.astype("float64") 
            out = 1/(1+np.exp(-tmp))
            yt = y[ii].reshape((o, -1))
            # for m in range(i):
                # for n in range(o): 
                    # dw[m][n] = lr*(yt[n]-out[n])*out[n]*(1-out[n])*xt[m]
            dw = lr * np.dot((yt-out)*out*(1-out), xt.reshape((-1, i))).T
            db = lr*(yt-out)*out*(1-out)
            w += dw 
            b += db 
    return w, b

i = 2
o = 2
lr = 0.1
w, b = model(x, y, lr, i, o)

for ii in range(10):
    print(1/(1+np.exp(-(np.dot(w.T, x[ii].reshape((i, -1)))))), y[ii])
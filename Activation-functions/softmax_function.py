#!/usr/bin/python3

import numpy as np

def softmax(x: np.array) -> np.array:
    """
        It's a normalized exponential function which takes n values as input and returns a probabilistic distribution of thses n values.

        Arguments
        ---------
            x: np.array

        Returns
        -------
            np.array
    """

    return np.exp(x)/sum(np.exp(x))
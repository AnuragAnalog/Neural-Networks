#!/usr/bin/python3

from math import exp

def sigmoid(a: float) -> float:
    """
        Rectified Linear Unit(ReLU), which has a ramp shape, it defines the positive part of the argument.

        Arguments
        ---------
            z: float value

        Returns
        -------
            Float value
    """

    return max(0, a)
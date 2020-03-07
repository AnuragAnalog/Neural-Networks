#!/usr/bin/python3

from math import log

def MSE(actual, predict):
    error = 0
    if type(actual) == type([]) or len(actual.shape) == 1:
        length = len(actual)
        for i in range(length):
            error += (actual[i] - predict[i])**2
    else:
        length = actual.shape[0] * actual.shape[1]
        for i in range(actual.shape[0]):
            for j in range(actual.shape[1]):
                error += (actual[i][j] - predict[i][j])**2
    error = error / length
    return error

def SSE(actual, predict):
    error = 0
    if type(actual) == type([]) or len(actual.shape) == 1:
        length = len(actual)
        for i in range(length):
            error += (actual[i] - predict[i])**2
    else:
        for i in range(actual.shape[0]):
            for j in range(actual.shape[1]):
                error += (actual[i][j] - predict[i][j])**2
    return error

def MAE(actual, predict):
    error = 0
    if type(actual) == type([]) or len(actual.shape) == 1:
        length = len(actual)
        for i in range(length):
            e = actual[i] - predict[i]
            if e > 0:
                error += e
            else:
                error += -1*e
    else:
        length = actual.shape[0] * actual.shape[1]
        for i in range(actual.shape[0]):
            for j in range(actual.shape[1]):
                e = actual[i][j] - predict[i][j]
                if e > 0:
                    error += e
                else:
                    error += -1*e
    error = error / length
    return error

def RMSE(actual, predict):
    error = 0
    if type(actual) == type([]) or len(actual.shape) == 1:
        length = len(actual)
        for i in range(length):
            error += (actual[i] - predict[i])**2
    else:
        length = actual.shape[0] * actual.shape[1]
        for i in range(actual.shape[0]):
            for j in range(actual.shape[1]):
                error += (actual[i][j] - predict[i][j])**2
    error = (error / length)**(1/2)
    return error

def NLL(actual, predict):
    error = 0
    if type(actual) == type([]) or len(actual.shape) == 1:
        length = len(actual)
        for i in range(length):
            e = actual[i] - predict[i]
            if e > 0:
                error += -1*log(e)
            else:
                error += -1*log(-1*e)
    else:
        length = actual.shape[0] * actual.shape[1]
        for i in range(actual.shape[0]):
            for j in range(actual.shape[1]):
                e = actual[i][j] - predict[i][j]
                if e > 0:
                    error += -1*log(e)
                else:
                    error += -1*log(-1*e)
    error = error / length
    return error
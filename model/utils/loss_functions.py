import numpy as np


def categorical_crossentropy(y_pred, y_true):
    return -np.sum(y_true * np.log(y_pred + 1e-15))


def categorical_crossentropy_derivative(y_pred, y_true):
    return y_pred - y_true

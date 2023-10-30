import numpy as np


def relu(z):
    return np.maximum(0, z)


def relu_derivative(z):
    return (z > 0).astype(float)


def softmax(z):
    exps = np.exp(z - np.max(z, axis=-1, keepdims=True))
    return exps / np.sum(exps, axis=-1, keepdims=True)


def sigmoid(z):
    return 1 / (1 + np.exp(-z))


def sigmoid_derivative(z):
    return sigmoid(z) * (1 - sigmoid(z))


def tanh(z):
    return (np.exp(z) - np.exp(-z)) / (np.exp(z) + np.exp(-z))


def tanh_derivative(z):
    return 1 - tanh(z) ** 2

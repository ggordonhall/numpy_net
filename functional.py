import numpy as np


def softmax(x):
    exps = np.exp(x - np.max(x))
    return exps / np.sum(exps, axis=1, keepdims=True)


def relu(x):
    return np.maximum(0, x)


def relu_derivative(dx, z):
    dz = np.array(dx, copy=True)
    dz[z <= 0] = 0
    return dz


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def sigmoid_derivative(dx, z):
    sig = sigmoid(z)
    return dx * sig * (1 - sig)

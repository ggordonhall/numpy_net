import numpy as np


def softmax(x):
    exp_scores = np.exp(x)
    return exp_scores / np.sum(exp_scores, axis=1, keepdims=True)


def relu(x):
    return np.maximum(0, x)


def relu_derivative(x):
    x[x <= 0] = 0
    x[x > 0] = 1
    return x


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def sigmoid_derivative(x):
    return x * (1.0 - x)


def tanh(x):
    return np.tanh(x)


def tanh_derivative(x):
    return 1.0 - np.tanh(x) ** 2


def cross_entropy(y_hat, y):
    n = y_hat.shape[0]
    return -np.sum(y * np.log(y_hat)) / n

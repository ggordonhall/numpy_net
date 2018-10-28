import numpy as np


def softmax(x):
    exps = np.exp(x - np.max(x))
    return exps / np.sum(exps, axis=1, keepdims=True)


def relu(x):
    return np.maximum(0, x)


def relu_derivative(x):
    x[x <= 0] = 0
    x[x > 0] = 1
    return x


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def sigmoid_derivative(x):
    return sigmoid(x) * (1 - sigmoid(x))


def cross_entropy(y_hat, y):
    """Calculate cross entropy loss function
    between predicted and actual probability
    distribution.

    Expression:
        1) L = -∑(y * log(ŷ))

    Arguments:
        y_hat {np.ndarray} --
            predicted probability distribution:
            (batch size * number of classes)
        y {np.ndarray} --
            indices of the correct class for each example
            in the batch: (batch size * 1)

    Returns:
        {float} -- average batch loss
    """

    bs = y.shape[0]
    log_likelihood = -np.log(y_hat[range(bs), y])
    return np.sum(log_likelihood) / bs


def cross_entropy_derivative(y_hat, y):
    """Calculate gradient of loss function
        with respect to the network output.

    Expression:
        1) dL = ŷ - y

    Arguments:
        y_hat {np.ndarray} --
            predicted probability distribution:
            (batch size * number of classes)
        y {np.ndarray} --
            indices of the correct class for each example
            in the batch: (batch size * 1)

    Returns:
        {np.ndarray} --
            gradients: (batch size * number of classes)
    """

    bs = y_hat.shape[0]
    y_hat[range(bs), y] -= 1
    return y_hat

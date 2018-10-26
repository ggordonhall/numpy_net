from typing import List, Tuple, Dict
import numpy as np

import functional as F
from data import Loader

loader = Loader("wine.csv", 2)
activations = {"relu": F.relu, "sigmoid": F.sigmoid}
derivatives = {"relu": F.relu_derivative, "sigmoid": F.sigmoid_derivative}


class NeuralNet:
    def __init__(self, inp_dim: int, out_dim: int, bs: int, lr: float, activation: str, *hidden_sizes: int):
        """Define network dimensions and activation functions.

        Arguments:
            inp_dim {int} -- the number of features
            out_dim {int} -- the number of classes
            bs {int} -- the batch size
            lr {float} -- the learning rate
            activation {str} -- name of the activation function
            hidden_sizes {int} -- variable number of hidden sizes
        """
        self._bs = bs
        self._lr = lr
        self._layers = [inp_dim, *hidden_sizes, out_dim]
        self._num_layers = len(self._layers) - 1
        self._params = self.init_parameters()

        if activation not in activations.keys():
            raise Exception("Activation function not supported!")
        self._activation = activations[activation]
        self._activation_derivative = derivatives[activation]

    def init_parameters(self):
        """Initialise network weights and biases and
        save in a dictionary: ``params``. Layers are
        1-indexed (input = layer 0).

        Weights initialised with a standard normal
        distribution: mean = 0, standard deviation = 1.
        Biases initialised with zeros.

        Parameters multiplied by 0.1 for numerical stability.

        Returns:
            {Dict[str, np.ndarray]} --
                parameter name to values mapping
        """
        np.random.seed(99)

        params = {}
        for idx in range(1, self._num_layers + 1):
            in_dim, out_dim = self._layers[idx - 1], self._layers[idx]
            standard_dist = np.random.randn(in_dim, out_dim)
            params["W" + str(idx)] = standard_dist * 0.1
            params["b" + str(idx)] = np.zeros((1, out_dim)) * 0.1
        return params

    def forward(self, x):
        """Forward propagate the input through the
        neural network. Return a probability distribution
        over classes. Save intermediate results for
        backpropagation in a cache.

        Arguments:
            x {np.ndarray} -- input (batch size * input dim)

        Returns:
            {Tuple[np.ndarray, Dict[str, np.adarry]]} --
                distribution and dict of intermediate values
        """

        cache = {}
        for idx in range(1, self._num_layers + 1):
            # cache output of previous layer
            cache["x" + str(idx - 1)] = x
            W = self._params["W" + str(idx)]
            b = self._params["b" + str(idx)]
            x, z = self.layer(x, W, b)
            # cache intermediate result
            cache["z" + str(idx)] = z
        return F.softmax(x), cache

    def layer(self, x, W, b):
        """Linear transformation followed by non-linear
        activation function, `g`.

        Expressions:
            1) z = xW + b
            2) a = g(z)

        Arguments:
            x {np.ndarray} --
                input (batch size * previous layer dim)
            W {np.ndarray} --
                weight matrix (previous layer dim * next layer dim)
            b {np.ndarray} --
                biases (batch size * next layer dim)

        Returns:
            {Tuple[np.ndarray]} -- a, z
        """
        z = x.dot(W) + b
        return self._activation(z), z

    def cross_entropy(self, y_hat, y):
        """Calculate cross entropy loss function
        between predicted and actual probability
        distribution.

        Loss function:
            L = - ∑ (y_i * log(y_hat_i))

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

        log_likelihood = -np.log(y_hat[range(self._bs), y])
        return np.sum(log_likelihood) / self._bs

    def cross_entropy_derivative(self, y_hat, y):
        """Calculate batch-average gradient of loss function
         with respect to the network output.

        Derivative:
            dL/dO = y_hat_i - y_i

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

        y_hat[range(self._bs), y] -= 1
        return y_hat / self._bs

    def backprop_layer(self, x, W, b, z, x_prev):
        """Calculate the gradients of a single
        feedforward layer.

        Gradients:
            dW = xT.dz
            db = ∑ dz / bs
            dx_prev = dz.WT

        Arguments:
            x {np.ndarray} -- g(z)
            W {np.ndarray} -- layer weights
            b {np.ndarray}  -- layer bias
            z {np.ndarray} -- z = x_prev * W + b
            x_prev {np.ndarray} --
                output of previous layer

        Returns:
            Tuple[np.ndarray] --
                gradients of the previous layer output,
                the weight matrix, and the bias with
                respect to the loss function.
        """

        dz = self._activation_derivative(x, z)
        dW = x_prev.T.dot(dz) / self._bs
        db = np.sum(dz, axis=0, keepdims=True) / self._bs
        dx_prev = dz.dot(W.T)
        return dx_prev, dW, db

    def backwards(self, y_hat, y, cache):
        """Calculate gradients of all parameters
        in the network.

        Arguments:
            y_hat {np.ndarray} --
                predicted probability distribution:
                (batch size * number of classes)
            y {np.ndarray} --
                indices of the correct class for each example
                in the batch: (batch size * 1)
            cache {Dict[str, np.ndarray]} --
                dict of intermediate values

        Returns:
            {Dict[str, np.ndarray]} --
                dict of gradients for each parameter
        """

        # dict of paramter gradients
        grads = {}
        # gradient of loss function with respect to net output
        dx_prev = self.cross_entropy_derivative(y_hat, y)
        # calculate gradients of layers in reverse
        for idx in reversed(range(1, self._num_layers + 1)):
            dx = dx_prev
            # get output of layer
            x_prev = cache["x" + str(idx - 1)]
            z = cache["z" + str(idx)]
            # get weights and biases of layer
            W = self._params["W" + str(idx)]
            b = self._params["b" + str(idx)]
            # calculate gradients of layer and store in `grads` dict
            dx_prev, dW, db = self.backprop_layer(dx, W, b, z, x_prev)
            grads["dW" + str(idx)] = dW
            grads["db" + str(idx)] = db

        return grads

    def parameter_update(self, grads, lr):
        """Update parameters of network according
        to backpropagated gradients and the learning
        rate.

        Arguments:
            {Dict[str, np.ndarray]} --
                dict of gradients for each parameter
            lr {float} -- the learning rate
        """

        for idx in range(1, self._num_layers + 1):
            self._params["W" + str(idx)] -= lr * \
                grads["dW" + str(idx)]
            self._params["b" + str(idx)] -= lr * \
                grads["db" + str(idx)]

    def train(self, n_steps):
        losses = []

        for step in range(n_steps):
            x, y = next(loader.iterator("train"))
            y = y.argmax(axis=1)
            y_hat, cache = self.forward(x)

            loss = self.cross_entropy(y_hat, y)
            losses.append(loss)

            grads = self.backwards(y_hat, y, cache)
            self.parameter_update(grads, self._lr)

        return self._params


net = NeuralNet(loader.num_features, loader.num_classes,
                2, 0.001, "relu", 20, 50, 20)
net.train(100)

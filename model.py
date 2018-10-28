from typing import List, Tuple, Dict

import math
import time
import logging
from copy import copy
import numpy as np

import functional as F
from utils import he_init
from utils import predictions
from utils import accuracy
from utils import plot_loss
from utils import time_since


class NeuralNet:
    def __init__(self, loader, lr, activation, loss, *hidden_sizes):
        """Define network dimensions and activation functions.

        Arguments:
            loader {data.Loader} -- a data loader
            lr {float} -- the learning rate
            activation {Tuple[Callable]} --
                tuple of activation function and its derivative
            loss {Tuple[Callable]} --
                tuple of loss function and its derivative
            hidden_sizes {int} -- variable number of hidden sizes
        """

        self._lr = lr
        self._loader = loader
        self._bs = loader.batch_size
        self._layers = [loader.num_features, *hidden_sizes, loader.num_classes]
        self._num_layers = len(self._layers) - 1

        self._activation, self._activation_derivative = activation
        self._loss, self._loss_derivative = loss

        self._params = self.init_parameters()

    def init_parameters(self):
        """Initialise network weights and biases and
        save in a dictionary: ``params``. Layers are
        1-indexed (input = layer 0).

        Weights initialised with He initialisation.
        Biases initialised with zeros.

        Returns:
            {Dict[str, np.ndarray]} --
                parameter name to values mapping
        """

        params = {}
        for idx in range(1, self._num_layers + 1):
            in_dim, out_dim = self._layers[idx - 1], self._layers[idx]
            params["W" + str(idx)] = he_init(in_dim, out_dim)
            params["b" + str(idx)] = np.zeros((1, out_dim))
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
            z = self.linear(x, W, b)
            # softmax activation for final layer
            activation = F.softmax if idx == self._num_layers else self._activation
            x = activation(z)
        return x, cache

    def linear(self, x, W, b):
        """Linear transformation of input.

        Expression:
            1) z = xW + b

        Arguments:
            x {np.ndarray} --
                input (batch size * previous layer dim)
            W {np.ndarray} --
                weight matrix (previous layer dim * next layer dim)
            b {np.ndarray} --
                biases (batch size * next layer dim)

        Returns:
            {np.ndarray} -- linear transformation of x
        """

        return np.dot(x, W) + b

    def backprop_layer(self, delta, x):
        """Calculate the gradients for a single
        feedforward layer.

        Gradients:
            dW = xT.δ
            db = ∑ (δ)

        Arguments:
            delta {np.ndarray} -- gradients of successive layer
            x {np.ndarray} -- output of previous layer

        Returns:
            {Tuple[np.ndarray]} --
                gradients of the the weight matrix, and
                the bias with respect to the loss function.
        """

        dW = np.matmul(x.T, delta)
        db = np.sum(delta, axis=0, keepdims=True)
        return dW, db

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

        grads = {}
        # gradient of loss function with respect to net output
        delta = self._loss_derivative(y_hat, y) / y_hat.shape[0]
        # calculate gradients of layers in reverse
        for idx in reversed(range(1, self._num_layers + 1)):
            # get intermediate output
            x = cache["x" + str(idx - 1)]
            # calculate grads
            dW, db = self.backprop_layer(delta, x)
            grads["dW" + str(idx)] = dW
            grads["db" + str(idx)] = db
            # first layer does not have an activation
            # function to differentiate
            if idx > 1:
                W = self._params["W" + str(idx)]
                delta = np.matmul(delta, W.T) * self._activation_derivative(x)

        return grads

    def update(self, grads, lr):
        """Update parameters of network according
        to backpropagated gradients and the learning
        rate.

        Arguments:
            {Dict[str, np.ndarray]} --
                dict of gradients for each parameter
            lr {float} -- the learning rate
        """

        params = copy(self._params)
        for idx in range(1, self._num_layers + 1):
            W_delta = lr * grads["dW" + str(idx)]
            b_delta = lr * grads["db" + str(idx)]
            params["W" + str(idx)] = params["W" + str(idx)] - W_delta
            params["b" + str(idx)] = params["b" + str(idx)] - b_delta
        return params

    def train(self, n_steps):
        """Run the training routine.

        Arguments:
            n_steps {int} -- the number of training iterations
        """

        losses = []
        report_every = math.ceil(n_steps * 0.01)

        start = time.time()
        logging.info("\n\nStarting training...\n\n")

        iter = self._loader.train_iterator(n_steps)
        for step, (x, y) in enumerate(iter):
            # get indices of batch labels
            y = predictions(y)
            # feedforward
            y_hat, cache = self.forward(x)
            if step % report_every == 0:
                # calculate loss
                loss = self._loss(y_hat, y)
                logging.info("Step: {}    Elapsed: {}    Loss: {:6g}".format(
                    step, time_since(start), loss))
                losses.append(loss)
            # calculate gradients and update parameters
            grads = self.backwards(y_hat, y, cache)
            self._params = self.update(grads, self._lr)

        logging.info("\n\nTraining complete!\n\n")
        plot_loss(losses, report_every)

    def test(self):
        """
        Run the evaluation routine.
        """

        accuracies = []
        logging.info("\n\nStarting evaluation...\n\n")

        for (x, y) in self._loader.test_iterator():
            # feedforward
            y_hat, _ = self.forward(x)

            # get indices of batch labels
            preds, y = predictions(y_hat), predictions(y)
            acc = accuracy(preds, y)
            accuracies.append(acc)

            logging.info(
                "Pred: {}   Actual: {}  Batch Accuracy: {:6g}".format(preds, y,  acc))

        average_accuracy = sum(accuracies) / float(len(accuracies))
        logging.info(
            "Average test batch accuracy: {}".format(average_accuracy))
        logging.info("\n\nEvaluation complete!\n\n")

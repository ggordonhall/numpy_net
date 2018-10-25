from typing import List

import numpy as np

import functional as F
from data import Loader

loader = Loader("wine.csv", 2)


class NeuralNet:
    def __init__(self, inp_dim: int, out_dim: int, batch_size: int, hidden_sizes: List[int]):
        """Initialise weights and biases for the network.

        Arguments:
            inp_dim {int} -- the number of features
            out_dim {int} -- the number of classes 
            batch_size {int} -- batch size
            hidden_sizes {List[int]} -- list of hidden sizes
        """

        self._h1, self._h2 = hidden_sizes

        # Weights (W) dim: (batch_size, inp_dim, h_dim)
        self._W1 = np.random.randn(batch_size, inp_dim, self._h1)
        self._b1 = np.zeros((batch_size, self._h2))

        self._W2 = np.random.randn(self._h1, self._h2)
        self._b2 = np.zeros((1, self._h2))

        self._W3 = np.random.randn(self._h2, out_dim)
        self._b3 = np.zeros((1, out_dim))

    def forward(self, x):
        print(x.shape)
        layer_1 = F.relu(self.affine(x, self._W1, self._b1))
        layer_2 = F.relu(self.affine(layer_1, self._W2, self._b2))
        layer_3 = self.affine(layer_2, self._W3, self._b3)
        return F.softmax(layer_3)

    def affine(self, x, W, b):
        return np.tensordot(x, W) + b

    def train(self, n_steps):
        for step in range(n_steps):
            x, y = next(loader.iterator("train"))
            y_hat = self.forward(x)
            loss = F.cross_entropy(y_hat, y)
            print(loss)
            break


net = NeuralNet(loader.num_features, loader.num_classes, 2, [20, 10])
net.train(10)

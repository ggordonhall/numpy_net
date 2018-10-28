import json

import functional as F
from data import Loader
from model import NeuralNet
from utils import set_logger

activations = {"relu": (F.relu, F.relu_derivative),
               "sigmoid": (F.sigmoid, F.sigmoid_derivative)}
losses = {"cross_entropy": (F.cross_entropy, F.cross_entropy_derivative)}


def main(config):
    data_path = config["paths"]["data"]
    loader = Loader(data_path, config["batch_size"])

    model_args = config["model"]
    activation_name = model_args["activation_function"]
    if activation_name not in activations.keys():
        raise Exception("Activation function not supported!")
    activation = activations[activation_name]

    loss_name = model_args["loss_function"]
    if loss_name not in losses.keys():
        raise Exception("Loss function not supported!")
    loss = losses[loss_name]

    net = NeuralNet(loader,
                    model_args["learning_rate"],
                    activation,
                    loss,
                    *model_args["hidden_layers"])

    net.train(config["num_steps"])
    net.test()


if __name__ == "__main__":
    config_path = "config.json"
    config = json.load(open(config_path, "r"))
    set_logger(config["paths"]["log"])

    main(config)

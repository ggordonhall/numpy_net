import json

from data import Loader
from model import NeuralNet
from utils import set_logger


def main(config):
    data_path = config["paths"]["data"]
    loader = Loader(data_path, config["batch_size"])

    model_args = config["model"]
    net = NeuralNet(loader, model_args["learning_rate"],
                    model_args["activation_function"], *model_args["hidden_layers"])

    net.train(config["num_steps"])
    net.test()


if __name__ == "__main__":
    config_path = "config.json"
    config = json.load(open(config_path, "r"))
    set_logger(config["paths"]["log"])

    main(config)

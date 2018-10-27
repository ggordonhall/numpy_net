import math
import time
import logging

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def predictions(dist: np.ndarray) -> np.ndarray:
    """
    Return index of maximum value of each row 
    in a batch.
    """
    return dist.argmax(axis=1)


def time_since(since: float) -> str:
    """Calculate time elapsed"""
    now = time.time()
    s = now - since
    m = math.floor(s / 60)
    s -= m * 60
    return "%dm %ds" % (m, s)


def plot_loss(losses, report_every):
    """Plot the training loss.

    Arguments:
        losses {List[float]} -- list of losses
        report_every {int} -- steps
    """

    series = pd.Series(losses)
    rolling = series.rolling(window=(report_every // 5))
    rolling_mean = rolling.mean()
    series.plot()
    rolling_mean.plot(color='red')

    plt.ylabel("Loss")
    plt.xlabel("Steps")
    plt.grid(True)
    plt.legend(("Training loss", "Running average"))
    plt.savefig("loss_plot.png", dpi=300)


def set_logger(log_path):
    """Set the logger to log info in terminal and file `log_path`.
    Args:
        log_path {str} -- where to log
    """
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    if not logger.handlers:
        # Logging to a file
        file_handler = logging.FileHandler(log_path, "w")
        file_handler.setFormatter(
            logging.Formatter("%(asctime)s:%(levelname)s: %(message)s")
        )
        logger.addHandler(file_handler)

        # Logging to console
        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(logging.Formatter("%(message)s"))
        logger.addHandler(stream_handler)

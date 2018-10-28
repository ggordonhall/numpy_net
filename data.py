from typing import List, Tuple, Union, Iterable

from random import shuffle, randint

import numpy as np
import pandas as pd


class BatchSizeError(Exception):
    def __init__(self):
        super().__init__("Batch size cannot exceed number of data pairs!")


class Loader:
    """Data preprocessing and loading class"""

    def __init__(self, file_path: str, batch_size: int=2, split_ratio: float=0.9):
        """Read and split data from ``file_path``.

        Arguments:
            file_path {str} -- path to csv file

        Keyword Arguments:
            batch_size {int} -- size of each batch (default: {2})
            split_ratio {float} -- train to test set size ratio (default: {0.9})
        """

        self._file_path = file_path
        self._batch_size = batch_size
        self._split_ratio = split_ratio

        pairs = self._read()
        shuffle(pairs)
        self._train_set, self._test_set = self._split(pairs)

    def _read(self) -> List[Tuple[np.ndarray]]:
        """Read data from csv at ``file_path``. Extract label and
        feature columns and convert them to np.ndarrays, then pair
        feature vectors with corresponding labels.

        Returns:
            {List[Tuple[np.ndarray]]} --
                a list of pairs of numpy features and labels
        """

        df = pd.read_csv(self._file_path)
        features = df.iloc[:, 1:].values.astype("float")
        labels = df.iloc[:, 0].values.astype("int")

        num_classes = len(set(labels))
        one_hot_labels = self._one_hot(labels, num_classes)
        features_norm = self._normalise(features)
        return list(zip(features_norm, one_hot_labels))

    def _split(self, pairs: List[Tuple[np.ndarray]]) -> Tuple[List[Tuple[np.ndarray]]]:
        """Split list of pairs into training and test
        sets according to the ratio ``split_ratio``.

        Arguments:
            pairs {List[Tuple[np.ndarray]]} -- 
                list of pairs of numpy features and labels

        Returns:
            {Tuple[List[Tuple[np.ndarray]]]} -- 
                tuple of training and test sets
        """

        split_idx = int(len(pairs) * self._split_ratio)
        return pairs[:split_idx], pairs[split_idx:]

    def _stack(self, batch: List[Tuple[np.ndarray]]) -> Tuple[np.ndarray]:
        """Stack lists of features and tuples into
        2-d numpy arrays.

        Arguments:
            batch {List[Tuple[np.ndarray]]} -- 
                list of (feature, label) tuples

        Returns:
            {Tuple[np.ndarray]} -- 
                2-d feature and label np.ndarrays
        """

        features = np.vstack([p[0] for p in batch])
        labels = np.vstack([p[1] for p in batch])
        return features, labels

    def _normalise(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply min-max normalisation on each dataframe column.
        Squashes values into the 0-1 range.

        Arguments:
            df {pd.DataFrame} -- columns of data

        Returns:
            {pd.DataFrame} -- normalised data
        """

        return (df - df.mean()) / (df.max() - df.min())

    def _one_hot(self, labels: List[int], num_classes: int) -> np.ndarray:
        """Convert label indicies to one-hot vectors.
        i.e. 3 -> [0, 0, 1]

        Arguments:
            labels {List[int]} -- list of labels
            num_classes: int -- number of classes

        Returns:
            {np.ndarray} -- one-hot matrix
        """

        indices = np.array(labels) - 1
        return np.eye(num_classes, dtype=int)[indices]

    def train_iterator(self, num_iters: int) -> Iterable:
        """Yield random chunks of data of size ``batch_size`` 
        from the training dataset.

        Arguments:
            num_iters {int} -- 
                number of generator iterations

        Raises:
            BatchSizeError --
                if batch_size > size of dataset

        Returns:
            {Iterable} -- a batched data iterator
        """

        pairs = self._train_set
        length = len(pairs)

        if self._batch_size > length:
            raise BatchSizeError()

        for _ in range(num_iters):
            idx = randint(0, length - self._batch_size - 1)
            batch = pairs[idx: idx + self._batch_size]
            yield self._stack(batch)

    def test_iterator(self) -> Iterable:
        """Iterate through the test data and yield chunks
        of size ``batch_size``.

        Raises:
            BatchSizeError --
                if batch_size > size of dataset

        Returns:
            {Iterable} -- a batched data iterator
        """

        pairs = self._test_set
        length = len(pairs)

        if self._batch_size > length:
            raise BatchSizeError()

        for idx in range(0, length, self._batch_size):
            batch = pairs[idx: min(idx + self._batch_size, length)]
            yield self._stack(batch)

    @property
    def num_features(self):
        return self._train_set[0][0].shape[0]

    @property
    def num_classes(self):
        return self._train_set[0][1].shape[0]

    @property
    def batch_size(self):
        return self._batch_size

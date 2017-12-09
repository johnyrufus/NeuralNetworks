#!/usr/bin/env python3
# algorithm.py : MLAlgorithm base class

import abc
import numpy as np

# Base class for all machine learning algorithms.
class MLAlgorithm:
    def __init__(self, source_file, model_file):
        self.source_file = source_file
        self.model_file = model_file
        self.p = 1

    def split_images_file(self, file):
        with open(file) as f:
            ncols = len(f.readline().split(' '))

        arr = np.loadtxt(file, usecols=range(1, ncols))
        orient, features = np.split(arr, [1], axis=1)
        nrows = int(features.shape[0] * self.p)
        return orient[0:nrows, :], features[0:nrows, :]

    @abc.abstractmethod
    def train(self):
        return

    @abc.abstractmethod
    def test(self):
        return

    def train_percent(self, p):
        self.p = p
        self.train()








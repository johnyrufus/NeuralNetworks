#!/usr/bin/env python3
# algorithm.py : Implement the local search algorithms

import abc
import numpy as np

# Base class for all machine learning algorithms.
class MLAlgorithm:
    def __init__(self, source_file, model_file):
        self.source_file = source_file
        self.model_file = model_file

    def split_images_file(self, file):
        with open(file) as f:
            ncols = len(f.readline().split(' '))

        arr = np.loadtxt(file, usecols=range(1, ncols))
        orient, features = np.split(arr, [1], axis=1)
        return orient, features

    @abc.abstractmethod
    def train(self):
        return

    @abc.abstractmethod
    def test(self):
        return








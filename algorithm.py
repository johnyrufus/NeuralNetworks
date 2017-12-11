#!/usr/bin/env python3
# algorithm.py : MLAlgorithm base class

import abc
import numpy as np
import pickle

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
        return orient, features

    @abc.abstractmethod
    def train(self):
        return

    @abc.abstractmethod
    def test(self):
        return

    def train_percent(self, p):
        self.p = p
        self.train()

    def save(self, obj, name):
        with open(name, 'wb') as f:
            pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

    def load(self, name):
        with open(name, 'rb') as f:
            return pickle.load(f)

    def get_image_names(self):
        with open(self.source_file):
            names = np.genfromtxt(self.source_file, dtype='str', usecols=range(0, 1))
        return names








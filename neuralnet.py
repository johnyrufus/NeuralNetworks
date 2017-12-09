#!/usr/bin/env python3
# neuralnet.py : Feed forward neural net and back propagation implementation

import numpy as np
from collections import defaultdict
import unittest

from algorithm import MLAlgorithm


class NeuralNet(MLAlgorithm):

    def __init__(self, source_file, model_file):
        self.layers_nodes = {0:192, 1:25, 2:4}
        self.layers = len(self.layers_nodes)
        self.weights = {i: {j: np.random.uniform(
            low=-1/np.sqrt(self.layers_nodes[i-1]), high=1/np.sqrt(self.layers_nodes[i-1]), size=self.layers_nodes[i-1])
            for j in range(self.layers_nodes[i])} for i in range(1, self.layers)}
        self.sigmoid = lambda x: 1 / (1 + np.exp(-x))
        self.mappings = {0: 0, 90: 1, 180: 2, 270: 3}
        super().__init__(source_file, model_file)

    def train(self):
        orient, features = self.split_images_file(self.source_file)
        print(orient.shape)
        print(features.shape)

        def derivative(x): return self.sigmoid(x) * (1 - self.sigmoid(x))

        orient = orient.ravel().tolist()

        for i in range(10):
            a = []
            y = []
            count = 0
            for j, feature in enumerate(features):
                feature = (feature - np.mean(feature)) / (np.std(feature))
                inputv = defaultdict(list)
                a.append(self.forward_prop(feature, self.layers, self.layers_nodes, self.weights, self.sigmoid, inputv))
                y.append([0.0] * 4)
                y[j][self.mappings[orient[j]]] = 1.0
                self.backward_prop(a[j], y[j], feature, self.layers, self.layers_nodes, self.weights, self.sigmoid, derivative, inputv)
                if a[j][2].index(max(a[j][2])) == self.mappings[orient[j]]:
                    count += 1

            print('Accuracy in training = {} after iteration {}'.format(count / features.shape[0], i))

            # Need to comment this out later, calling here, to see how the iteration affects the test data accuracy
            self.test()

    def forward_prop(self, feature, layers, layers_nodes, weights, activation, inputv):
        a = defaultdict(list)

        a[0] = feature
        inputv[0].append(feature)
        for l in range(1, layers):
            for node in range(layers_nodes[l]):
                inputv[l].append(np.dot(weights[l][node], a[l - 1]))
                a[l].append(activation(inputv[l][node]))
        return a

    def backward_prop(self, a, y, feature, layers, layers_nodes, weights, activation, derivative, inputv):

        delta = defaultdict(list)
        delta[layers-1] = [derivative(inputv[layers-1][j]) * (y[j]-a[layers-1][j]) for j in range(layers_nodes[layers-1])]

        for layer in range(layers-2, 0, -1):
            delta[layer] = [0] * layers_nodes[layer]
            for i in range(layers_nodes[layer]):
                total = 0
                for j in range(layers_nodes[layer + 1]):
                    total += weights[layer+1][j][i] * delta[layer + 1][j]
                delta[layer][i] = derivative(inputv[layer][i]) * total

        for layer in range(layers-1):
            for i in range(layers_nodes[layer]):
                for j in range(layers_nodes[layer + 1]):
                    weights[layer+1][j][i] = weights[layer+1][j][i] + 0.1 * a[layer][i] * delta[layer+1][j]

    def test(self):
        test_file = 'test-data.txt'
        orient_test, features_test = self.split_images_file(test_file)
        orient_test = orient_test.ravel().tolist()
        count = 0
        for i, feature in enumerate(features_test):
            feature = (feature - np.mean(feature)) / (np.std(feature))
            a = self.forward_prop(feature, self.layers, self.layers_nodes, self.weights, self.sigmoid, defaultdict(list))

            if a[2].index(max(a[2])) == self.mappings[orient_test[i]]:
                count += 1
        print('Accuracy in testing = {} '.format(count / features_test.shape[0]))


class NeuralNetTest(unittest.TestCase):
    def test(self):
        layers = 3
        layers_nodes = {0: 3, 1: 2, 2: 1}
        weights = {1: {0: np.array([0.5, 0.5, 0.5]),
                       1: np.array([0.1, 0.2, 0.3]), },
                   2: {0: np.array([0.1, 0.2, ]), }}
        feature = np.array([2, 3, 4, ])

        train_file = 'train-data.txt'
        model_file = 'model_file_nn.txt'
        nn = NeuralNet(train_file, model_file)

        inputv = defaultdict(list)
        a = nn.forward_prop(feature, layers, layers_nodes, weights, lambda x: x, inputv)
        print(a)

        self.assertCountEqual(a[0], [2, 3, 4])
        self.assertCountEqual(a[1], [4.5, 2.0])
        self.assertCountEqual(a[2], [0.85000000000000009])

        nn.backward_prop(a, [1], feature, layers, layers_nodes, weights, lambda x: x, lambda x: x, inputv)



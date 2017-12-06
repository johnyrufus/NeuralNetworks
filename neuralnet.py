#!/usr/bin/env python3
# neuralnet.py : Feed forward neural net and back propagation implementation

import numpy as np
from collections import defaultdict
import unittest

from algorithm import MLAlgorithm


class NeuralNet(MLAlgorithm):

    def train(self):
        orient, features = self.split_images_file(self.source_file)
        print(orient.shape)
        print(features.shape)

        def sigmoid(x): return 1 / (1 + np.exp(-x))

        layers_nodes = {0:192, 1:50, 2:4}
        layers = len(layers_nodes)
        # weights = [ np.random.uniform(low=-1/np.sqrt(layers_nodes[i-1]), high=1/np.sqrt(layers_nodes[i-1]),
        # size=layers_nodes[i-1]) for i in range(layers-1)]

        weights = {i: {j: np.random.uniform(
            low=-1/np.sqrt(layers_nodes[i-1]), high=1/np.sqrt(layers_nodes[i-1]), size=layers_nodes[i-1])
            for j in range(layers_nodes[i])} for i in range(1, layers)}
        # print(weights)
        # features = np.hstack([features, np.ones([features.shape[0], 1], int)])
        print(features.shape)
        # a = {i: np.ones((1,layers_nodes[i]), dtype=np.float) for i in range(0, layers-1)}
        orient = orient.ravel().tolist()
        print(orient)

        for i in range(2):
            a = []
            for j, feature in enumerate(features):
                #print(j)
                inputv = defaultdict(list)
                a.append(self.forward_prop(feature, layers, layers_nodes, weights, sigmoid, inputv))

    def forward_prop(self, feature, layers, layers_nodes, weights, activation, inputv):
        a = defaultdict(list)

        a[0] = feature
        inputv[0].append(feature)
        for l in range(1, layers):
            for node in range(layers_nodes[l]):
                inputv[l].append(np.dot(weights[l][node], a[l - 1]))
                a[l].append(activation(inputv[l][node]))

            #a[l] = [activation(np.dot(weights[l][node], a[l - 1])) for node in range(layers_nodes[l]-1)]
            '''if l != layers-1:
                a[l].append(1)
                inputv[l].append(1)'''
        return a

    def backward_prop(self, a, y, feature, layers, layers_nodes, weights, activation, derivative, inputv):

        delta = defaultdict(list)

        '''for j in range(layers_nodes[layers - 1]-1):
            delta[layers - 1] = []
            inputv[layers - 1]
            print(inputv[layers - 1], layers-1, j)
            inputv[layers - 1][j]
            derivative(inputv[layers - 1][j]) * 1
            y[j] - a[layers - 1][j]'''


        delta[layers-1] = [derivative(inputv[layers-1][j]) * (y[j]-a[layers-1][j]) for j in range(layers_nodes[layers-1])]
        #print(delta)

        for layer in range(layers-2, 0, -1):
            #print(delta)
            #print(weights)
            delta[layer] = [0] * layers_nodes[layer]
            for i in range(layers_nodes[layer]):
                total = 0
                for j in range(layers_nodes[layer + 1]):
                    #print(i, j)
                    total += weights[layer+1][j][i] * delta[layer + 1][j]
                delta[layer][i] = derivative(inputv[layer][i]) * total

        for layer in range(layers-1):
            for i in range(layers_nodes[layer]):
                for j in range(layers_nodes[layer + 1]):
                    weights[layer+1][j][i] = weights[layer+1][j][i] + 0.1 * a[layer][i] * delta[layer+1][j]

        #print(delta)

    def test(self):
       pass

class NeuralNetTest(unittest.TestCase):
    def test(self):
        layers = 3
        layers_nodes = {0: 3, 1: 2, 2: 1}
        weights = {1: {0: np.array([0.5, 0.5, 0.5]),
                       1: np.array([0.1, 0.2, 0.3]), },
                   # 2:np.array([0.5, 0.5, 0.5, 0.5]),
                   # 3:np.array([0.5, 0.5, 0.5, 0.5])},
                   2: {0: np.array([0.1, 0.2, ]), }}
        # 1:np.array([0.5, 0.5, 0.5, 0.5]),
        # 2: np.array([0.1, 0.5, 0.5, 0.5])}}
        feature = np.array([2, 3, 4, ])

        train_file = 'train-data.txt'
        model_file = 'model_file_nn.txt'
        nn = NeuralNet(train_file, model_file)

        a = nn.forward_prop(feature, layers, layers_nodes, weights, lambda x: x, defaultdict(list))
        print(a)

        self.assertCountEqual(a[0], [2, 3, 4])
        self.assertCountEqual(a[1], [4.5, 2.0])
        self.assertCountEqual(a[2], [0.85000000000000009])



train_file = 'train-data.txt'
model_file = 'model_file_nn.txt'
nn = NeuralNet(train_file, model_file)
nn.train()

# Run test every time

nn_test = NeuralNetTest()
nn_test.test()

#!/usr/bin/env python3
# neuralnet.py : Feed forward neural net and back propagation implementation

import numpy as np
from collections import defaultdict

from algorithm import MLAlgorithm


class NeuralNet(MLAlgorithm):

    def train(self):
        orient, features = self.split_images_file(self.source_file)
        print(orient.shape)
        print(features.shape)

        def sigmoid(x): return 1 / (1 + np.exp(-x))

        layers_nodes = {0:192, 1:10, 2:4}
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
                a.append(self.forward_prop(feature, layers, layers_nodes, weights, sigmoid))

    def forward_prop(self, feature, layers, layers_nodes, weights, activation):
        a = defaultdict(list)

        a[0] = feature
        for l in range(1, layers):
            for node in range(layers_nodes[l]):
                a[l].append(activation(np.dot(weights[l][node], a[l - 1])))

            '''a[l] = [activation(np.dot(weights[l][node], a[l - 1])) for node in range(layers_nodes[l]-1)]
               if l != layers-1:
               a[l].append(1)
            '''
        return a


    def test_forward_prop(self):
        layers = 3
        layers_nodes = {0: 3, 1: 2, 2: 1}
        weights = {1: {0:np.array([0.5, 0.5, 0.5]),
                       1:np.array([0.1, 0.2, 0.3]),},
                       #2:np.array([0.5, 0.5, 0.5, 0.5]),
                       #3:np.array([0.5, 0.5, 0.5, 0.5])},
                   2: {0:np.array([0.1, 0.2,]),}}
                       #1:np.array([0.5, 0.5, 0.5, 0.5]),
                       #2: np.array([0.1, 0.5, 0.5, 0.5])}}
        feature = np.array([2, 3, 4,])
        a = self.forward_prop(feature, layers, layers_nodes, weights, lambda x: x)
        print(a)
        assert a[0] == [2,3,4]
        assert a[1] == [4.5, 2.0]
        assert a[2] == [0.85000000000000009]


    def test(self):
       pass


train_file = 'train-data.txt'
model_file = 'model_file_nn.txt'
nn = NeuralNet(train_file, model_file)
nn.test_forward_prop()

#!/usr/bin/env python3
# orient.py : base orient class

from knn import KNN
from adaboost import Adaboost
from neuralnet import NeuralNet
import numpy as np

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import sys


def main():

    if len(sys.argv) < 5:
        print('Usage: ')
        print('./orient.py [train/test] train/test_file.txt model_file.txt [model]')
        sys.exit()

    (train_or_test, source_file, model_file, model) = sys.argv[1:5]
    print(train_or_test, source_file, model_file, model)

    if model == 'nearest':
        algorithm = KNN
    elif model == 'adaboost':
        algorithm = Adaboost
    else:
        algorithm = NeuralNet

    if train_or_test == 'train':
        algorithm(source_file, model_file).train()
    else:
        algorithm(source_file, model_file).test()


def experiment_knn():
    pass

def experiment_adaboost():
    pass

def experiment_neuralnet():
    train_file = 'train-data.txt'
    model_file = 'model_file_nn.txt'

    trainacc = []
    testacc = []
    x = []

    for i in range(10, 151, 10):
        nn = NeuralNet(train_file, model_file)
        nn.iterations = 3
        nn.layers_nodes[1] = i
        nn.weights = {i: {j: np.random.uniform(
            low=-1 / np.sqrt(nn.layers_nodes[i - 1]), high=1 / np.sqrt(nn.layers_nodes[i - 1]),
            size=nn.layers_nodes[i - 1])
            for j in range(nn.layers_nodes[i])} for i in range(1, nn.layers)}
        x.append(i)
        trainacc.append(nn.train())
        testacc.append(nn.test())


    plt.figure()
    plt.plot(x, trainacc, '--r', label='Training Accuracy')
    plt.plot(x, testacc, '--b', label='Testing Accuracy')
    plt.title('Accuracy vs Hidden Nodes')
    plt.xlabel('Hidden Nodes')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.savefig('AccuracyHiddenNodes.png', dpi=100)


if __name__ == '__main__':
    main()

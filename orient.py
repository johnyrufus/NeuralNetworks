#!/usr/bin/env python3
# orient.py : base orient class

from knn import KNN
from adaboost import Adaboost
from neuralnet import NeuralNet

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


if __name__ == '__main__':
    main()

def experiment_knn():
    pass

def experiment_adaboost():
    pass

def experiment_neuralnet():
    train_file = 'train-data.txt'
    model_file = 'model_file_nn.txt'
    nn = NeuralNet(train_file, model_file)
    nn.train()
    nn.test()




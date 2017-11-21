#!/usr/bin/env python3
# orient.py : base orient class

from knn import KNN


def main():
    train_file = 'train-data.txt'
    model_file = 'model_file_knn.txt'
    knn = KNN(train_file, model_file)
    knn.train()

    test_file = 'test-data.txt'
    knn = KNN(test_file, train_file)
    knn.test()


if __name__ == '__main__':
    main()


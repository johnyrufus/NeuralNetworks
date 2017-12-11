#!/usr/bin/env python3
# knn.py : KNN implementation

import numpy as np
from algorithm import MLAlgorithm
from shutil import copyfile

class KNN(MLAlgorithm):

    def __init__(self, source_file, model_file):
        self.k = 20
        super().__init__(source_file, model_file)

    def train(self):
        copyfile(self.source_file, self.model_file)

    def test(self):

        orient_train, features_train = self.split_images_file(self.model_file)
        orient_test, features_test = self.split_images_file(self.source_file)

        def predict(test_image_row):
            dists = np.sum(np.square(test_image_row - features_train), axis=1)
            ret = orient_train[np.argsort(dists)[0:self.k]]
            ret = ret.ravel()
            uniqw, inverse = np.unique(ret, return_inverse=True)
            counts = np.bincount(inverse)
            max_res = uniqw[np.argmax(counts)]
            return max_res

        predicted = np.apply_along_axis(predict, 1, features_test)
        original = orient_test.ravel()
        comps = predicted == original
        print('Accuracy of prediction: ', np.count_nonzero(comps)/comps.shape[0])

        output_file = open('output.txt', "w")
        with output_file:
            for name, guess in zip(self.get_image_names(), predicted):
                output_file.write(name + ' ' + str(guess) + '\n')


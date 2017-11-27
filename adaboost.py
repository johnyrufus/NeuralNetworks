''' This code is still being modified before it can be integrated with the main class ```

import numpy as np
from math import log

def split_images_file(file):
    with open(file) as f:
        ncols = len(f.readline().split(' '))

    arr = np.loadtxt(file, usecols=range(1, ncols))
    orient, features = np.split(arr, [1], axis=1)
    return orient, features

# Read in the data as an array in the form [(orientation), (pixels)] for each image
file = 'train-data.txt'
training_array = split_images_file(file)
total_training_obs = len(training_array[0])

'''
Methodology: 
(1) Loop through all of the pixels in the test image - find those where i < j where i and j are individual pixels
(2) Filter by different orientations in the test images, find the pairs that generate the most True
'''

# Filter by images for a certain rotation to find out how often that pixel equality holds in the training set
def filter_by_rot(training_array, rot):
    img_rot = []
    for i,x in enumerate(training_array[0][:]):
        if x == rot:
            img_rot.append(training_array[1][i])
    return np.array(img_rot)

# When choosing stumps, use this function to update the weights for observations that already selected classifiers
# did not perform well with by overweighting those incorrect observations - all weights are normalized to sum to 1
def update_weights(current_weights, classifier_chosen_array, error_rate):

    un_normalized_weights = np.array([error_rate if i == 1 else i for i in current_weights])
    new_weights = np.multiply(current_weights,un_normalized_weights)
    normalized_weights = new_weights/sum(new_weights)

    return normalized_weights

# *** IN PROGRESS *** currently using orientation = 90 for code testing - will eventually incorporate 0, 90, 180, 270
# Go through the training set by a certain rotation and count the instances when one pixel column is less than another
# pixel column --> find the highest frequencies to decide what the stumps should be

orientation_array = filter_by_rot(training_array, 90)
(rot_rows, pixels) = orientation_array.shape

num_stumps = 4
# Start with an equal weight for each pixel combination
classifier_weight = np.array([1/rot_rows for obs in range(rot_rows)])
results = []

# Store the pixel combinations that yielded the most True
stumps_chosen = []

# Start collecting the number of stumps desired
for i in range(num_stumps):

    best_score = -np.inf
    best_pair = (-9999, -9999)
    best_outcome = -9999

# Evaluate each pixel in the training images to find which are most important for each type of rotated image
    for i in range(pixels):
        for j in range(pixels):
            # Don't compare a column to itself or if the stump has already been selected/stored
            if i != j and (i,j) not in stumps_chosen and (j, i) not in stumps_chosen:
                # Find out the result of each pixel column comparsion as an integer (1 = T, 0 = F) within this rotation
                num_true = np.less(orientation_array[: , i], orientation_array[: , j]).astype(int)
                # Find the weighted score of each pair/classifier
                weighted_score = np.sum(np.multiply(num_true, classifier_weight))
                # Find the weighted total of each combination and locate the highest scoring pair
                # Note: simple comparison is used for speed - adding to dict & finding the minimum is nearly 8x slower
                if weighted_score > best_score:
                    best_score = weighted_score
                    best_pair = (i, j)
                    best_outcome = num_true

    # Add the highest score to the collected list of stumps for this rotation
    stumps_chosen.append(best_pair)
    # Since we filtered the data to a particular orientation, these are deemed to be correct
    error_rate = (total_training_obs - rot_rows)/(total_training_obs)
    # Redefine the weights used in the scoring
    classifier_weight = update_weights(classifier_weight, best_outcome, error_rate)
    # Q: How to figure out the weight for each classifier under a given orientation? Based off frequency now - doesn't seem 
    # right since each within an orientation will have the same weight.  In progress...
    print(best_pair, 0.5*log((1-error_rate)/error_rate))

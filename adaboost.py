import numpy as np
from math import log
from collections import defaultdict

def split_images_file(file):
    with open(file) as f:
        ncols = len(f.readline().split(' '))

    arr = np.loadtxt(file, usecols=range(1, ncols))
    orient, features = np.split(arr, [1], axis=1)
    return orient, features

# Filter by images for a certain rotation to find out how often that pixel equality holds in the training set
def filter_by_rot(training_array, rot):
    img_rot = []
    for i,x in enumerate(training_array[0][:]):
        if x == rot:
            img_rot.append(training_array[1][i])
    return np.array(img_rot)

# When choosing stumps, use this function to update the weights for observations that already selected classifiers
# did not perform well with by overweighting those incorrect observations - all weights are normalized to sum to 1
def update_weights(current_weights, error_rate):

    un_normalized_weights = np.array([error_rate if i == 1 else i for i in current_weights])
    new_weights = np.multiply(current_weights,un_normalized_weights)
    normalized_weights = new_weights/sum(new_weights)

    return normalized_weights

# *** IN PROGRESS *** currently using orientation = 90 for code testing - will eventually incorporate 0, 90, 180, 270
# Go through the training set by a certain rotation and count the instances when one pixel column is less than another
# pixel column --> find the highest frequencies to decide what the stumps should be
def get_weak_classifiers_and_errors(training_array, orientation, num_stumps):
    orientation_array = filter_by_rot(training_array, orientation)
    # print('Orientation array for', orientation, orientation_array)
    (rot_rows, pixels) = orientation_array.shape

    # Start with an equal weight for each pixel combination
    classifier_weight = np.array([1/rot_rows for obs in range(rot_rows)])
    results = []

    # Store the pixel combinations that yielded the most True
    stumps_chosen = []
    # Store the errors to be able to make a vote for each stump
    errors = []
    correct = []

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
                    num_true = np.greater(orientation_array[: , i], orientation_array[: , j]).astype(int)
                    # Find the weighted score of each pair/classifier
                    # print(np.sum(np.multiply(num_true, classifier_weight)))
                    weighted_score = np.sum(np.multiply(num_true, classifier_weight))
                    # Find the weighted total of each combination and locate the highest scoring pair
                    # Note: simple comparison is used for speed - adding to dict & finding the minimum is nearly 8x slower

                    if weighted_score > best_score:
                        best_score = weighted_score
                        best_pair = (i, j)
                        best_outcome = num_true
                        # Since we filtered the data to a particular orientation, these are deemed to be correct
                        error_rate = (rot_rows - sum(num_true)) / rot_rows
                        # print('Best score',best_score, 'Best Pair',best_pair, 'Error rate',error_rate, 'Num True', sum(num_true), 'Rows', rot_rows)
                        # print('New error rate for orientation', orientation,error_rate)

        # Redefine the weights used in the scoring
        classifier_weight = update_weights(classifier_weight, error_rate)

        # Add the highest score to the collected list of stumps for this rotation
        stumps_chosen.append(best_pair)
        # print(error_rate)
        errors.append(0.5*log((1-error_rate)/error_rate))
        num_true = np.sum(num_true)
        # print(stumps_chosen, best_pair, error_rate))
    # return (stumps_chosen, errors)
    #     print('STUMP FINISHED with these results',num_true, rot_rows, stumps_chosen, errors)
    return stumps_chosen, errors

# Start looking at the test data
def vote_on_image(image_array, classifier_info, num_stumps):

    orientations = [0,90,180,270]
    orient_votes = defaultdict(float)

    # print('This is the image array', image_array)
    for k,v in classifier_info.items():
        votes = []
        # print(orientation, get_weak_classifiers_and_errors(training_array, orientation, num_stumps))
        # weak_classifiers = get_weak_classifiers_and_errors(training_array, orientation, num_stumps)
        classifiers = v[0][0]
        errors = v[0][1]
        # print(classifiers, errors)
        for classifier in classifiers:
            col1, col2 = classifier[0], classifier[1]
            if image_array[col1] > image_array[col2]:
                votes.append(1)
            else:
                votes.append(0)
        # print(errors, votes)
        orient_votes[k] = sum(np.multiply(votes, errors))

    # print(orient_votes)
    return max(orient_votes, key = orient_votes.get)

# Read in the data as an array in the form [(orientation), (pixels)] for each image
print('Reading in the training data...')
file = 'train-data.txt'
training_array = split_images_file(file)
total_training_obs = len(training_array[0])
print('Training data read successfully!\n')
'''
Methodology: 
(1) Loop through all of the pixels in the test image - find those where i < j where i and j are individual pixels
(2) Filter by different orientations in the test images, find the pairs that generate the most True
'''

all_stump_results = []

for s in range(20):
    print('Beginning to train the model...')
    orientations = [0,90,180,270]
    stumps = s
    classifier_dict = defaultdict(list)
    for orientation in orientations:
        weak_classifiers = get_weak_classifiers_and_errors(training_array,orientation,stumps)
        classifier_dict[orientation].append(weak_classifiers)
    print('Model trained.')

    image_file = 'test-data.txt'
    test_orientations = split_images_file(image_file)[0]
    images = split_images_file(image_file)[1]

    guesses = []
    for image in images:
       guesses.append(vote_on_image(image,classifier_dict, stumps))

    guesses = np.array(guesses)

    total_correct = 0
    for i,j in zip(guesses, test_orientations):
        if i == j:
            total_correct += 1
    percent_correct = total_correct/len(guesses)
    print('% Correct', percent_correct)
    all_stump_results.append(percent_correct)

print(all_stump_results)

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedShuffleSplit

import os
import glob

### Model Utils  ###

# Returns a smoothed angle across the previous and predicted angles.
def smooth(new_angle, angle, weights = [0.75, 0.25]):
    return np.average([new_angle, angle], weights = weights)

# Finds the nearest element to the specified value
# - Returns: index of nearest element
def find_nearest(array, value):
    return np.argmin(np.abs(array - value))

# Evenly splits the samples into a train / test set.
# - Returns: 4D tuple of train/test samples (x, y)
def stratified_split(X, y, split = 0.7, splits = 1):
    strat_split = StratifiedShuffleSplit(n_splits = splits, test_size = (1.0 - split))
    for train, test in strat_split.split(X, y):
        trainx_samples, trainy_samples = X[train], y[train]
        testx_samples, testy_samples = X[test], y[test]
    
    return (trainx_samples, trainy_samples, testx_samples, testy_samples)

# Categorises the labels to a 1 of K encoding using the supplied range values.
def categorise_labels(y, ranges):
    ranges = np.array(ranges)
    Y_res = np.empty(shape=(len(y), len(ranges)))
    for i in range(len(y)):
        Y_res[i, find_nearest(ranges, y[i])] = 1
    return Y_res

# One-hot encodes the Y label (1 x N) array into N x K one hot matrix. 
def one_hot_encode(labels, nb_classes):
    one_hot = np.zeros((len(labels), nb_classes))
    for y in labels:
        one_hot[y, int(y)] = 1
    return one_hot

# One hot encoding, for converting actions (zero based) into
# N x M encoded label matrix.
def one_hot_encode(y, no_classes):
    one_hot = np.zeros([len(y), no_classes])
    for i in np.unique(y):
        one_hot[y == i, int(i)] = 1
    return one_hot

# Shuffles the input arrays randomly
def shuffle(arr):
    rand = np.random.choice(len(arr), len(arr), replace=False)
    return np.array(arr)[rand]

# Randomly samples a batch from the input
def sample(X, y, sample_size):
    args = np.random.choice(len(X), sample_size, replace=False)
    return np.array(X)[args], np.array(y)[args]

# Returns an array of checkpoint tuples for filenames and their loss, sorted by loss in ascending order.
def get_checkpoints(path = "./model/checkpoint-*.h5"):
    files = glob.glob(path)
    checkpoints = []
    if (len(files) > 0):
        for checkpoint in files:
            name = os.path.splitext(checkpoint.split("\\")[1])[0]
            epoch = int(name.split("-")[1])
            loss = float(name.split("-")[2])
            checkpoints.append((checkpoint, epoch, loss))
        checkpoints = pd.DataFrame(checkpoints)
        checkpoints = checkpoints.sort([2, 1], ascending = [True, False])
    return np.array(checkpoints)

# Keras layer popping function
## Intermediate fix for issues (#2371, #2640)
def pop_layer(model):
    if not model.outputs:
        raise Exception('Model cannot be popped: model is empty.')

    model.layers.pop()
    if not model.layers:
        model.outputs = []
        model.inbound_nodes = []
        model.outbound_nodes = []
    else:
        # fix model outputs
        model.outputs = [model.layers[-1].output]
        model.output_layers = [model.layers[-1]]
        model.layers[-1].outbound_nodes = []
    # reset model status
    model.built = False
    # return model
    return model

### Import Libraries ###
import numpy as np
import pandas as pd
import densenet as d
import kanet as k
import road as rd
import generator as g
import modelutils as mut

import cv2
import csv
import os

from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold

# import Keras modules
from keras.models import Model
from keras.callbacks import Callback, ModelCheckpoint, ReduceLROnPlateau, EarlyStopping, CSVLogger
from keras.optimizers import Adam, RMSprop, SGD
from keras import backend as K 
from callbacks import EarlyStoppingByDivergence

from keras.models import load_model

import modelconfig as config

### Model Initialisation ###

# Creates and trains a feature detector using a generator and training params
def Pretrain(train_generator, train_length, validation_generator, validation_length, class_ranges, img_dim=(320, 160, 3), \
            resize_factor = 0.5, layers = 40, growth = 12, bottleneck = True, reduction = 0.5, noise = 0, dropout = 0.8,
            loss='categorical_crossentropy', optimizer = 'adam', metrics = ['accuracy'], epochs = 50, callbacks = None):
    # number of labels
    num_classes = len(class_ranges)
    # initial epoch
    epoch = 0
    # find previous checkpoints
    checkpoints = mut.get_checkpoints(config.checkpoint_pattern)

    # Initialize a new DenseNet model
    # CK 170321
    # refactored DenseNet creation for use with drive and qlearner
    #(input, output, classifier) = d.DenseNet(img_dim, resize_factor, nb_classes=num_classes, depth=layers, growth_rate=growth, 
    #                    bottleneck=bottleneck, reduction=reduction, dropout_rate=dropout, noise=noise)
    (input, output, classifier) = k.KaNet(img_dim, resize_factor, nb_classes=num_classes, dropout_rate=dropout, noise=noise)
    model = Model(input = input, output = classifier)
    # Compile the model using the supplied params
    print("Initialized.")
    print("Compiling...")
    model.compile(optimizer, loss, metrics=metrics, loss_weights=None, sample_weight_mode=None)
    print(" - Done.")
    
    if (len(checkpoints) > 0):
        print("Restoring weights from previous checkpoint...")
        checkpoint = checkpoints[0, :]
        epoch = int(checkpoint[1])
        model.load_weights(checkpoint[0])
        print("\n - Done.  (Checkpoint loss = {}, epoch = {})".format(checkpoint[2], epoch))
    
    print("## Model ##\n{}".format(model.summary()))
    print("Training...")
    # Train the model using the supplied generator
    history = model.fit_generator(train_generator, samples_per_epoch = train_length, \
                                    validation_data = validation_generator, nb_val_samples = validation_length, \
                                    nb_epoch = epochs, callbacks = callbacks, initial_epoch = epoch)
    
    return model

### Run ###

# Define parameters used in training
class_ranges = np.array(config.class_ranges)

train_file = os.path.join(config.feature_training_dir, "training_data.csv")
test_file = os.path.join(config.feature_training_dir, "testing_data.csv")

# setup callbacks, save every checkpoint, check for overfitting and reduce learning rate as required
callbacks = [
    CSVLogger("./model/training-loss.csv", separator=',', append = True),
    #ReduceLROnPlateau(monitor='val_loss', factor = 0.1, cooldown = 0, patience = 2, min_lr = 10e-6), # if using non-adaptive optimizer
    EarlyStopping(monitor='val_acc', min_delta = 0.0001, patience = 7),
    EarlyStoppingByDivergence(patience = 7),
    ModelCheckpoint(filepath = config.checkpoint_pattern.replace("*","{epoch:02d}-{val_loss:.2f}"), monitor='val_loss', save_best_only=True, save_weights_only=True, verbose=0),
]

# initialise datasets
X_train, y_train, X_test, y_test = None, None, None, None

# load previously saved samples if available
if (os.path.exists(train_file) and os.path.exists(test_file)):
    print("Initialising datasets...")
    # load datasets
    train_samples = np.array(pd.read_csv(train_file, sep=','))
    test_samples = np.array(pd.read_csv(test_file, sep=','))
    # extract into training / testing matrices
    X_train = train_samples[:, 0]
    y_train = train_samples[:, 1].astype(np.uint8)
    X_test = test_samples[:, 0]
    y_test = test_samples[:, 1].astype(np.uint8)
    print(" - Done.\n")
else:
    # load samples for feeding to generators
    print("Constructing datasets...")
    samples = []
    with open(config.feature_training_file) as csvfile:
        reader = csv.reader(csvfile)
        # skip header
        if (config.feature_training_skip_header):
            next(reader) 
        # read samples
        for sample in reader:
            # is relative path
            if (config.feature_training_image_relative): 
                name = os.path.join(config.feature_training_dir, sample[config.feature_training_image_col].split('/')[-1])
            else: name = sample[config.feature_training_image_col]
            
            if os.path.exists(name) and len(sample[config.feature_training_steering_col]) > 0:
                samples.append([name, mut.find_nearest(class_ranges, float(sample[config.feature_training_steering_col]))])

    ### Input sampling ###
    samples = np.array(samples)
    X = samples[:, 0]
    y = samples[:, 1].astype(np.uint8)
    # train / test split
    X_train, y_train, X_test, y_test = mut.stratified_split(X, y, config.train_split)

    # compress samples into shape for saving
    train_samples, test_samples = np.column_stack((X_train, y_train)), np.column_stack((X_test, y_test))
    # save datasets for resuming training
    with open(train_file, mode = "wb") as file:
        np.savetxt(file, train_samples, delimiter=",", fmt = "%s")
    with open(test_file, mode = "wb") as file:
        np.savetxt(file, test_samples, delimiter=",", fmt = "%s")
    print(" - Done.\n")

### Training Params ###


print("{} Samples loaded.".format(len(X_train)))
print("Training size = {}, Validation size = {}".format(len(X_train), len(X_test))) 

print("Initializing generators...")

# compile and train the model using the generator function
train_generator = g.generator(X_train, y_train, class_ranges, batch_size = config.batch_size, noise = config.noise)
validation_generator = g.generator(X_test, y_test, class_ranges, batch_size = config.batch_size, noise = 0)
print(" - Done.\n")

### Optimiser ###
#optimizer = SGD(lr = config.learning_rate, momentum = config.momentum, decay = config.weight_decay, nesterov = config.use_nesterov)
#optimizer = Adam(lr = config.learning_rate)
optimizer = RMSprop(lr = config.learning_rate)

print("Initializing network...")

### Training ###
# pretrain feature detector network
model = Pretrain(train_generator, config.epoch_train_samples, validation_generator, config.epoch_test_samples, class_ranges, img_dim = config.img_dim, 
                 resize_factor = config.resize_factor, layers = config.layers, growth = config.growth_rate, bottleneck = config.bottleneck, reduction = config.reduction, 
                 optimizer = optimizer, dropout = config.dropout, epochs = config.max_epochs, noise = config.noise, callbacks = callbacks)

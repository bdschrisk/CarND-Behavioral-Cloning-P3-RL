from keras.models import Model
from keras.layers import Input, merge
from keras.layers.core import Dense, Activation, Flatten, Dropout, Lambda
from keras.layers.advanced_activations import PReLU
from keras.layers.convolutional import Convolution2D, Cropping2D
from keras.layers.pooling import MaxPooling2D
from keras.layers.noise import GaussianNoise
from keras.layers.pooling import AveragePooling2D, MaxPooling2D, GlobalAveragePooling2D
from keras.layers.normalization import BatchNormalization
from keras.regularizers import l2

import keras.backend as K

# import custom keras layers
from extensions.layers.core_extended import Resize

### Model Functions ###

# Normalizes the input using min-max normalization.
def Normalizer(x):
    return (x / 255) - 0.5

def Normalizer_shape(input_shape):
    return input_shape

# Returns an input tensor for the rest of the network
# - input_shape: image shape shape
# - crop_width: 2D tuple of cropping from left and right edges
# - crop_height: 2D tuple of cropping from top and bottom edges
# - resize_factor: scaling factor to apply to the input after cropping
# - sigma: Scalar value of the gaussian noise function.
def InputLayer(input_shape, crop_width = (0,0), crop_height = (50,20), resize_factor = 0.5, sigma = 0.1):
    # Initialise input network
    input = Input(shape=input_shape)
    # Reshape for Cropping layer
    #model = Reshape((input_shape[2], input_shape[1], input_shape[0]))(input)
    # Apply cropping
    model = Cropping2D(cropping=(crop_height, crop_width))(input)
    # Sample wise min-max normalization layer
    model = Lambda(Normalizer, output_shape=Normalizer_shape)(model)
    # Calculate new dimensions after cropping
    new_height = (input_shape[0] - (crop_height[0] + crop_height[1])) * resize_factor
    new_width = (input_shape[1] - (crop_width[0] + crop_width[1])) * resize_factor
    # add Resize layer (CUSTOM)
    model = Resize((int(new_height), int(new_width)), axis=(1, 2), interpolation='nearest_neighbor')(model)
    
    if (sigma > 0):
        # Apply noise
        model = GaussianNoise(sigma)(model)
    
    return (input, model)

def KaNet(img_dim, resize_factor, nb_classes, dropout_rate=None, weight_decay=1E-4, noise = 0, verbose=True):
    (input, output) = InputLayer(img_dim, resize_factor = resize_factor, sigma = 0)
    # first convolutional layer
    output = Convolution2D(64, 5, 5, border_mode='valid')(output)
    # downsampling layer (avg pool)
    output = AveragePooling2D((4, 4), strides=(2, 2))(output)
    output = Activation('relu')(output)
    
    # second convolutional layer
    output = Convolution2D(32, 3, 3, border_mode='valid')(output)
    # downsampling layer (avg pool)
    output = AveragePooling2D((2, 2), strides=(1, 1))(output)
    output = Activation('relu')(output)

    # third convolutional layer
    output = Convolution2D(16, 2, 2, border_mode='valid')(output)
    # downsampling layer (avg pool)
    output = AveragePooling2D((2, 2), strides=(1, 1))(output)
    output = Activation('relu')(output)
    
    # spatial features layer
    output = AveragePooling2D((2, 2), strides=(2, 2))(output)
    output = Flatten()(output)
    
    # Hidden layer 1
    output = Dense(256, W_regularizer=l2(weight_decay), b_regularizer=l2(weight_decay))(output)
    output = PReLU()(output)
    output = Dropout(dropout_rate)(output)
    # Hidden layer 2
    output = Dense(128, W_regularizer=l2(weight_decay), b_regularizer=l2(weight_decay))(output)
    output = PReLU()(output)
    output = Dropout(dropout_rate)(output)
    
    # classification output layer
    model = Dense(nb_classes, activation='softmax', W_regularizer=l2(weight_decay), b_regularizer=l2(weight_decay))(output)
    
    return (input, output, model)
### Data Augmentation / Generation ###

# Import libraries
import numpy as np
import matplotlib.image as mpimg
import cv2
import time
import os
import random

import modelutils as mut

import logging
logging.basicConfig(filename='./data/training.log',level=logging.DEBUG)

# Gets a uniform random from the interval 0-1
def random_uniform(min = 0.0, max = 1.0):
    return min + (np.random.uniform() * (max - min))

# Noise function
def add_noise(x, noise = 0.2, channel = 2):
    if (noise == 0):
        return x
    rand = (0.5 + random_uniform(0., noise))
    x[:,:,channel] = x[:,:,channel] * rand
    x = np.clip(x, 0, 255).astype(np.uint8)
    return x

# Adds noise to an image
def add_image_noise(img, noise):
    aug_image = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    aug_image = add_noise(aug_image, noise)
    aug_image = cv2.cvtColor(aug_image, cv2.COLOR_HSV2RGB)
    return aug_image

# Gets a minibatch from the entire sample for the given population
def get_minibatch(samples, label, offset, per_batch):
    result = samples[samples[:, 1] == label][offset:][0:per_batch].tolist()
    if (len(result) < per_batch):
        diff = per_batch - len(result)
        population = samples[samples[:, 1] == label].tolist()
        while (diff > 0 and len(population) > 0):
            result = result + random.choice(population)
            diff -= 1
    return result

# Flips the angle taking into account the range categories
def flip_angle(angle, ranges):
    return mut.find_nearest(ranges, -ranges[angle])

def generator(X, y, ranges, batch_size = 32, noise = 0.3, flip = True):    
    # setup
    per_batch = int(round(batch_size / len(ranges), 0))
    logging.info("Sampling: batch size = {}, sample size = {}".format(batch_size, per_batch))
    
    # count samples
    num_samples = len(y)
    errors = 0
    logging.info("-----------------------------")
    logging.info("\tStart time: " + time.strftime('%X %x %Z'))
    logging.info("Samples: - \n{}".format(X[0:3]))
    
    # do generator
    while 1:
        (X, y) = mut.sample(X, y, len(y))
        
        for offset in range(0, num_samples, batch_size):
            images = []
            angles = []
            augmented_images = []
            augmented_angles = []
            center_image = None
            
            batch_x, batch_y = X[offset:offset+batch_size], y[offset:offset+batch_size]
            # add sample (with noise) aswell as augmented versions
            for batch_sample in range(len(batch_x)):
                try:
                    logging.info("Processing sample => '{}'\t\t= {}".format(batch_x[batch_sample], batch_y[batch_sample]))
                    center_image = mpimg.imread(batch_x[batch_sample])
                    center_angle = int(batch_y[batch_sample])
                    
                    images.append(center_image)
                    angles.append(center_angle)
                    
                    # augment image
                    if (noise > 0):
                        aug_image = add_image_noise(center_image, noise)
                        augmented_images.append(aug_image)
                        augmented_angles.append(center_angle)
                    
                    if (flip):
                        # add rotated version
                        flip_image = np.fliplr(center_image)
                        augmented_images.append(flip_image)
                        augmented_angles.append(flip_angle(center_angle, ranges))
                        
                        if (noise > 0):
                            noise_flip_image = add_image_noise(flip_image, noise)
                            augmented_images.append(noise_flip_image)
                            augmented_angles.append(flip_angle(center_angle, ranges))
                except Exception as ex:
                    errors += 1
                    logging.warning("-- Error occurred while processing file '{}'\n - {}".format(batch_sample[0], ex))
            # concat data
            images += augmented_images
            angles += augmented_angles
            
            if (len(images) < batch_size and len(images) > 0):
                logging.warning("-- Repeating images as not enough data.\n- Images: [{}]".format(["'{}',".format(b) for b in batch_x]))
                while (len(images) < batch_size):
                    idx = np.random.randint(0, len(images))
                    center_image = images[idx]
                    center_angle = angles[idx]
                    if (noise > 0):
                        center_image = add_image_noise(center_image, noise)
                    elif (flip):
                        center_image = add_image_noise(center_image, noise)
                        center_image = np.fliplr(center_image)
                        center_angle = flip_angle(center_angle, ranges)
                    images.append(center_image)
                    angles.append(center_angle)
            # prepare inputs
            X_train = np.array(images)
            y_train = np.array(angles)
            # one-hot encoding of labels
            y_train = mut.one_hot_encode(y_train, len(ranges))
            # return batch of correct size
            X_train, y_train = X_train[0:batch_size], y_train[0:batch_size].astype(np.float32)
            
            logging.info("## Batch [{}] ##\nX [{}] =\n{}\n\nY =\n{}".format(
                        int(offset/batch_size), 
                        X_train.shape, 
                        [" - '{}'\n".format(b) for b in batch_x], 
                        y_train))
            
            yield (X_train, y_train)
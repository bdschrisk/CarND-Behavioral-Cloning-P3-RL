import argparse
import base64
import json

import numpy as np
import socketio
import eventlet
import eventlet.wsgi

from PIL import Image
from PIL import ImageOps
from flask import Flask, render_template
from io import BytesIO

import time
from datetime import datetime

import os
import cv2
import csv

from keras.models import Model, model_from_json
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array
from keras.models import load_model

# import custom Keras layers
from extensions.layers.core_extended import Resize

# import custom modules
import densenet as dn
import kanet as kn
from qmodel import QModel
import modelutils as mut
import modelconfig as config

# CK 170307: Removed as using Theano...
# Fix error with Keras and TensorFlow
#import tensorflow as tf
#tf.python.control_flow_ops = tf

sio = socketio.Server()
app = Flask(__name__)
model = None

online_train = True
prev_state = None

class_ranges = np.array(config.class_ranges)
max_speed = 30
cruise_throttle = 0.1

frame = 0
use_reinforcement = True

@sio.on('telemetry')
def telemetry(sid, data):
    # define globals
    global prev_state
    global frame
    global cruise_throttle
    
    if (data is not None):
        # The current steering angle of the car
        prev_angle = mut.find_nearest(class_ranges, float(data["steering_angle"]))
        steering_angle = class_ranges[prev_angle]
        # The current throttle of the car
        throttle = float(data["throttle"])
        # The current speed of the car
        speed = float(data["speed"])
        
        # The current image from the center camera of the car
        image = Image.open(BytesIO(base64.b64decode(data["image"])))
        image_array = np.asarray(image)
        
        (reward, qval) = (np.nan, np.nan)
        # get predicted action
        if (use_reinforcement):
            # compute reward
            (action, qval) = model.predict(image_array)
            reward = model.compute_reward(image_array, speed / max_speed, max_speed = 1.0)
            if (online_train and (frame % config.experience_skip_count == 0)):
                # Update Q
                model.update(prev_state, image_array, prev_angle, reward, verbose = 1)
        else:
            action = np.argmax(model.predict(image_array[None,:,:,:])[0])
        
        # map angle from label
        p_angle = class_ranges[action]
    
        # smooth angle
        steering_angle = mut.smooth(p_angle, steering_angle)
        throttle = cruise_throttle
    
        guessed = ("*" if np.isnan(qval) else "")
        print("Steering angle = {:.2f}  (Throttle: {:.2f}, Speed: {:.2f})  |  (Action = {}{}, Angle = {:.3f})  |  R = {}"
                                .format(steering_angle, throttle, speed, action, guessed, p_angle, reward))
        # temporal state / action pair
        
        prev_state = image_array
        frame += 1
        # send control commands
        send_control(steering_angle, throttle)
        # save frame if recording output
        if args.output != '':
            timestamp = datetime.utcnow().strftime('%Y_%m_%d_%H_%M_%S_%f')[:-3]
            image_filename = os.path.join(args.output, timestamp)
            image.save('{}.jpg'.format(image_filename))


@sio.on('connect')
def connect(sid, environ):
    print("connect ", sid)
    send_control(0, 0)


def send_control(steering_angle, throttle):
    sio.emit("steer", data={
        'steering_angle': steering_angle.__str__(),
        'throttle': throttle.__str__()
    }, skip_sid=True)

def run_replay(samples_file):
    ### Experience Replay ###
    print("Running experience replay...")
    print("(This may take some time)")
    # load sequential samples
    X, y, s = [], [], []
    # load replay data
    with open(samples_file) as csvfile:
        reader = csv.reader(csvfile)
        #next(reader) # skip header
        for sample in reader:
            #name = os.path.join("./data/IMG/", sample[0].split('/')[-1])
            name = sample[0]
            if os.path.exists(name) and len(sample[3]) > 0:
                X.append(np.asarray(Image.open(name)))
                y.append(mut.find_nearest(class_ranges, float(sample[3])))
                s.append(float(sample[6])/max_speed)
    # subset experiences
    X = np.array(X)[0:config.experience_count]
    y = np.array(y)[0:config.experience_count]
    s = np.array(s)[0:config.experience_count]
    # do experience replay
    model.experience_replay(X, y, s, memory_length = config.experience_memory_length, skip_count = config.experience_skip_count)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Remote Driving')
    parser.add_argument('modelpath', type=str, help='Path to KaNet model weights file.')
    parser.add_argument('output', type=str, nargs='?', default='', help='Path to image folder. This is where the images from the run will be saved.')
    parser.add_argument('--reinforce', type=bool, nargs='?', help='Enabled online learning during autonomous mode')
    parser.add_argument('--replay', type=bool, nargs='?', help='Runs experience replay on a new Q model (ignored if a Q model already exists)')
    parser.add_argument('--ludicrous', type=bool, nargs='?', help='Enables ludicrous mode!')
    parser.add_argument('--norl', type=bool, nargs='?', help='Disables reinforcement learning altogether and relies on the underlying KaNet model')
    parser.set_defaults(reinforce = False, ludicrous = False, replay = False, norl = False)
    args = parser.parse_args()
    
    # define params
    num_classes = len(config.class_ranges)
    
    print("\n### Autonomous Driving Project ###")
    print("")
    
    # define drive params
    cruise_throttle = (0.8 if args.ludicrous == True else cruise_throttle)
    online_train = args.reinforce
    # Indicates whether models exist
    q_valid = det_valid = False
    
    # check for existing q model
    q_valid = os.path.exists(os.path.join("./model/", QModel.model_name()))
    
    # if model weights passed in, override and initialise new QModel    
    # otherwise find previous Q model
    det_valid = (args.modelpath != None and os.path.exists(args.modelpath))
    model_path = args.modelpath
    # check for existing Q model if not overriding
    q_valid = ((det_valid == False) and (os.path.exists(os.path.join(config.model_path, QModel.model_name()))))
    
    if (q_valid and not args.norl):
        print("QModel found. Restoring...")
        
        # Workaround: keras save / load model method not working for the entire Q Net...
        (input, output, classifier) = kn.KaNet(config.img_dim, config.resize_factor, nb_classes=num_classes, dropout_rate=config.dropout, noise=config.noise)
        model = QModel.create(input, output, class_ranges, layers = config.Q_layers, prune = config.Q_prune)
        model = QModel.load(model, config.model_path)
        # reset learning switch
        if (online_train): model.reinforce()  
        else: model.offline()
        
        print(" - Initialized.  Online learning: {}, throttle factor: {}".format(online_train, cruise_throttle))
    else:
        # if no model weights passed, then load the best checkpoint from training
        #if not det_valid: model_path = mut.get_checkpoints(config.checkpoint_pattern)[0,0]
        online_train = True # guarantee Q Learning enabled
        
        print("\nInitialising model:\n - weights file = '{}', reinforcement = {}, throttle = {}".format((model_path if det_valid else 'None*'), online_train, args.ludicrous))
        
        # Workaround: use Net builder to construct previous model and load weights
        # then use the actual inputs and output layers and pass into QNetwork builder
        # ideally we'd load_model then trim the last layers to avoid hardcoding model creation.
        (input, output, classifier) = kn.KaNet(config.img_dim, config.resize_factor, nb_classes=num_classes, dropout_rate=config.dropout, noise=config.noise)
        model = Model(input = input, output = classifier)
        # Lock layers if there is a previously trained model
        lock_trainable_layers = False
        # Load weights if found, otherwise create a new model ready for online learning if using reinforcement learning
        if (os.path.exists(model_path)):
            print("\nRestoring model... ")            
            model.load_weights(model_path)
            print(" - Model restored.")
            # we have weights, lets lock'n'roll...
            lock_trainable_layers = True
        
        if (model != None and args.norl == False):
            print("\nCreating new QModel...")
            
            # Create QNetwork from initial
            qnetmodel = QModel.create(input, output, class_ranges, layers = config.Q_layers, prune = config.Q_prune, lock_layers = lock_trainable_layers)
            model = QModel(qnetmodel, class_ranges, gamma = config.Q_gamma, min_reward = config.Q_minreward, max_reward = config.Q_maxreward,
                                epsilon = config.Q_epsilon, epoch = 0, interval = config.Q_interval, sprint = config.Q_sprint)
            
            print(" - Done.")
        
        if (args.replay):
            run_replay("./data/REPLAY/driving_log.csv")
    
    use_reinforcement = (args.norl == False)
    if (online_train and use_reinforcement):
        cruise_throttle = config.online_learning_throttle # we're still learning...
    
    print("")
    if (args.output != ''):
        if (not os.path.exists(args.output)):
            print("Creating output directory...")
            os.makedirs(args.output)
        print("Saving output to: {}\n".format(args.output))
    
    # wrap Flask application with engineio's middleware
    app = socketio.Middleware(sio, app)

    # deploy as an eventlet WSGI server
    eventlet.wsgi.server(eventlet.listen(('', 4567)), app)
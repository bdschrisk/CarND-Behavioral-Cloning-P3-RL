import road as rd
import numpy as np
import os
import pickle
import random

from keras.models import Model
from keras.layers.core import Dense, Activation, Flatten
from keras.optimizers import Adam, RMSprop, SGD
from keras.models import load_model
import keras.backend as K

from keras.callbacks import Callback, LambdaCallback, CSVLogger

from extensions.metrics import qmax

import modelutils as mut

### QModel (Deep Q Network) object ###

class QModel(object):
    def __init__(self, model, actions, gamma = 0.3, min_reward = -10, max_reward = 100, epsilon = 0.8,
                 eps_annealed = 0.8, epoch = 0, interval = 500, sprint = 50, iteration = 1):
        """ Initialises a new QModel object with the supplied Q Network model.
        
        # Arguments
            model: a keras Deep Q network
            gamma: discount factor for delayed rewards, higher values favour delayed rewards over immediate ones.
            min_reward: Minimum reward value
            max_reward: Maximum reward value
            actions: array of possible actions
            epsilon: epsilon greedy parameter to induce randomness into the action selection function
            eps_annealed: annealed epsilon parameter
            epoch: current epoch or 1 for a new QModel
            interval: number of epoch intervals between checkpoints
            sprint: length of each learning sprint
            iteration: current iteration, used in resuming runs
        """
        self.QModel = model
        # Init reward parameters
        self.gamma = gamma
        self.min_reward = min_reward
        self.max_reward = max_reward
        self.actions = actions
        self.epsilon = epsilon
        self.eps_annealed = eps_annealed
        self.sprint = sprint
        self.iteration = iteration
        self.epoch = epoch
        self.interval = interval
        self.learning = False
        self.Q = None
        self.Qtm1 = None
        self.reward = 0
        
        self.callbacks = [CSVLogger(filename = "./model/qlearner-loss.csv", append = True)]
    
    @staticmethod
    def create(input, output, actions, layers = [1024, 128], activation = 'relu', prune = 1, lock_layers = True):
        """ Constructs a new QLearning model using the trained Keras model, the network is pruned and converted into 
            a new Deep-Q Network with the specified possible actions and size of the QLearning layers 
            
            Arguments:
             - input: Keras input layer object
             - output: Keras output layer object instance
             - actions: array of possible actions
             - layers: array of Q layers to be added in the output layers
             - activation: activation function in layers
             - prune: number of layers to prune prior to adding Q layers (ignored)...
             - lock_layers: prevents previously created layers from updating weights in training
        """
        # define output size
        output_size = len(actions)
        # Trims the last layer and appends a Q-Learning layer
        # Lock weights in previous layers (transfer learning) 
        
        # Trim last layer(s) using custom function as layers.pop() not working...
        #for layer in range(prune):
        #    model = mut.pop_layer(model)
        
        # Append Q Layers
        # get network output as input into QNetwork
        # output = model.layers[-1]
        
        # make sure network is flat
        if (len(K.int_shape(output)) >= 3):
            output = Flatten()(output)
        
        # create DeepQ hidden layers
        for size in layers:
            output = Dense(size, init='lecun_uniform')(output)
            output = Activation(activation)(output)
        
        # add final output layer
        output = Dense(output_size)(output)
        # using a linear activation as our q-value action estimator is continuous (as per paper)
        output = Activation('linear')(output)
        
        # init model using previous models input layer
        #input = network.layers[0]
        
        # optimise using RMSProp
        optimizer = RMSprop(lr = 1e-4)
        qnetmodel = Model(input = input, output = output)
        qnetmodel.compile(loss='mse', optimizer = optimizer, metrics = ['mae', qmax])
        
        # lock weights in the feature detector network
        if (lock_layers):
            for l in range(len(qnetmodel.layers) - ((len(layers) + 1) * 2)):
                qnetmodel.layers[l].trainable = False
        
        print("\nDeepQNet Model:\n".format(qnetmodel.summary()))
        
        # return the Q network
        return qnetmodel
    
    def offline(self):
        """ Turns off online learning """
        self.iteration = self.sprint + 1
        self.learning = False
    
    def reinforce(self):
        """ Turns on reinforcement learning """
        self.iteration = 1
        self.learning = False
        self.eps_annealed = self.epsilon
    
    def compute_reward(self, sample, speed, max_speed = 100, verbose = 1):
        """ Reward computation function """
        
        # compute edge distances
        try:
            # TODO: Fix this as it's not accurate enough to be useful in training
            [left_dx, right_dx] = rd.find_edge_distances(sample, rd.get_road_options())
            dx = (np.sum(left_dx) - np.sum(right_dx)) / (np.sum(left_dx) + np.sum(right_dx))
            sse = np.sum(dx ** 2.)
            rval = (sse * self.min_reward) + ((1.0-sse) * self.max_reward)
            if (verbose > 0):
                print(" - Cross-track error signal: {}".format(sse))
        except:
            rval = self.min_reward
        # clip reward
        rval = np.clip(rval, self.min_reward, self.max_reward)
        
        return rval
    
    def sprinting(self):
        """ Returns a boolean indicating whether the model is currently during a learning sprint """
        return self.iteration <= self.sprint and self.iteration >= 1
    
    def predict(self, sample):
        """ Predicts the next action from the given input and returns a 2D tuple of action / q value pair """
        # store previous Q
        if (self.Q is not None):
            self.Qtm1 = np.copy(self.Q)
        
        if (((np.random.rand() < self.eps_annealed) and self.sprinting()) or self.learning):
            action = np.random.randint(0, len(self.actions))
            qval = np.nan
        else:
            self.Q = self.QModel.predict(sample[None,:,:,:], batch_size = 1)[0]
            action = np.argmax(self.Q)
            qval = self.Q[action]
        
        # if not sprinting then progress iteration
        if (not self.sprinting()):
            self.iteration += 1./self.sprint
        
        return (action, qval)
    
    def update(self, sample_tm1, sample, action, reward, verbose = 1, use_sprint = True):
        """ Updates the Q model using the current sample as input and previous state and action/value pair at (t-1) """
        reward = 0
        
        assert sample_tm1 is not None, "sample_tm1 cannot be none"
        assert sample is not None, "sample cannot be none"
        
        # check if sprinting or forcing an update
        if ((not self.learning and self.sprinting()) or (use_sprint == False)):
            if ((self.Q is not None) and (self.Qtm1 is not None)):
                # get reward of the current state
                self.reward = np.average([reward, self.reward])
                # check for valid reward
                if (not np.isnan(self.reward)):
                    self.learning = True
                    
                    oldQ = self.Qtm1[action]
                    # get current max Q
                    (maxQ, argQ) = (np.max(self.Q), np.argmax(self.Q))
                    # Q update rule: Q(s, a) = reward + gamma * Q_max(s', a')
                    newQ = self.reward + (self.gamma * maxQ)
                    self.Qtm1[action] = newQ
                    
                    # update Q w.r.t previous state and new Q value update
                    self.QModel.fit(sample_tm1[None,:,:,:], self.Qtm1[None,:], batch_size = 1, nb_epoch = 1, initial_epoch = self.epoch, callbacks = self.callbacks, verbose = 0)
                    
                    # if interval, then save checkpoint
                    if (self.iteration % self.interval == 0):
                        self.epoch += 1
                        self.save()
                    # epsilon annealing
                    if (self.eps_annealed > 0.1):
                        self.eps_annealed -= self.epsilon/self.sprint
                    # log output
                    if (verbose >= 1):
                        print("Updated Q:\n  - angle = {}, reward = {:.4f}, Q [{:.3f}] => [{:.3f}]".format(action, self.reward, oldQ, newQ))
                    # increase iteration, if converged then wait until next sprint
                    self.iteration += 1
                    if (use_sprint and (self.iteration > self.sprint)):
                        self.iteration = 0 # end of sprint
                        self.eps_annealed = self.epsilon # reset epsilon
                    
                    self.learning = False
            
        #return self.reward
    
    def experience_replay(self, X, y, s, memory_length = 50, batch_size = 32, skip_count = 4, save_after = True):
        """ Runs experience replay over the inputs on the network 
            Arguments:
                - X: training samples
                - y: indexed labels (0..n == 0 <= len(actions))
                - memory_length: length of the memory buffer
                - batch_size: batch size of each update
                - save_after: saves the model after running experiences
        """
        # set offline during experience replay
        self.offline()
        # run experiences
        memory = []
        t = 0
        r = []
        # cache rewards for efficiency
        for i in range(0, len(y)-1):
            rval = self.compute_reward(X[i + 1], s[i], max_speed = 1., verbose = 0)
            r.append(rval)
        
        for i in range(0, len(y) - 1, skip_count):
            # valid step sizes
            if (i + 1) >= len(y):
                break
            
            # get starting state
            (state, action, state_p, reward) = (X[i], y[i], X[i + 1], r[i])
            
            # create memories
            if (len(memory) < memory_length):
                memory.append((state, action, state_p, reward))
            else:
                # replay experiences
                if (t < (memory_length - 1)): t += 1
                else: t = 0
                # overwrite current memory
                memory[t] = (state, action, state_p, reward)
                
                # sample a batch from memory
                batch = random.sample(memory, batch_size)
                
                # iterate through each sample
                for sample in batch:
                    # update Q w.r.t state and state prime, given the action taken
                    (state, action, state_p, reward) = sample
                    # predict Q states, sets Q[t-1] for update
                    actionq_tm1 = self.predict(state)
                    # predict new state, sets current Q for update
                    actionq = self.predict(state_p)
                    # update Q at t-1
                    self.update(state, state_p, action, reward, use_sprint = False, verbose = 0)
        
        # save the model
        if (save_after):
            self.save()
        
        # finish with reinforcement enabled
        self.reinforce()
    
    def save(self, path = "./model/"):
        """ Saves the Q Learning model to the path """
        
        # write object properties
        props = { "actions":self.actions, "gamma":self.gamma, "min_reward":self.min_reward, "max_reward":self.max_reward, 
                  "epsilon":self.epsilon, "eps_annealed":self.eps_annealed, "epoch":self.epoch, "interval": self.interval, 
                  "sprint":self.sprint, "iteration":self.iteration, "Q":self.Q, "Qtm1":self.Qtm1 }
        with open(os.path.join(path, QModel.model_name()), mode = "wb") as f:
            pickle.dump(props, f)
        
        self.QModel.save_weights(os.path.join(path, "QNetwork.h5"))
    
    @staticmethod
    def load(qnetmodel, path = "./model/"):
        """ Loads the weights for the Q Learning model from the path and returns a restored QModel object """
        qnetmodel.load_weights(os.path.join(path, "QNetwork.h5"))
        # restore object properties
        with open(os.path.join(path, QModel.model_name()), mode = "rb") as f:
            props = pickle.load(f)
        
        qmodel = QModel(qnetmodel, actions = props["actions"], gamma = props["gamma"], min_reward = props["min_reward"], max_reward = props["max_reward"],
                        epsilon = props["epsilon"], eps_annealed = props["eps_annealed"], epoch = props["epoch"], interval = props["interval"], 
                        sprint = props["sprint"], iteration = props["iteration"])
        qmodel.Q = props["Q"]
        qmodel.Qtm1 = props["Qtm1"]
        # return model
        return qmodel
    
    @staticmethod
    def model_name():
        """ Returns the name of the file used when saving """
        return "QNetwork.p"
        
        
    



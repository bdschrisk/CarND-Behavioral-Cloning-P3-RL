# Behavioural Cloning Project
### "Project Karl"
*__"My first customer was a lunatic. My second had a death wish."__*

#### NOT FINISHED...

## Overview
This project is intended to apply behavioural cloning using Deep-Q Networks for the purposes of controlling a simulated car driving around a track within a simulator.

In 2013, the Google DeepMind team showed that it was possible for an agent to learn directly from its environment using Q-Learning and deep neural networks. Likewise, in this project I intend to show that a self driving car can teach itself how to drive on a simple track using Deep-Q Networks and computer vision as feedback.

### Reinforcement Learning
In order for Reinforcement learning to work, a reward function is required so that the agent can measure the outcome of its actions within the environment.  The reward function is defined as the following:

Reward function: 
$$ f_{reward}(s) = (f_{loss} * r_{min}) + (1 - f_{loss}) * r_{max} $$ 
$$ f_{loss}(dx) = \sum{(left_{dx}-right_{dx})} \over {\sum{left_{dx}} + \sum{right_{dx}}} ^ 2 $$

The **left_dx** and **right_dx** parameters refer to the distance between the car and neighbouring edges - including road edges, which is the computer vision part.  The min and max refer to the range of the reward function, I used __-20__ to __100__.  Clipping the reward values prevents the gradients from exploding and corrupting learned weights.

Essentially, this reward function is designed so that the car learns to drive in the middle of the road.  With further work this approach can easily be extended to other road edges and markers such as vehicles, pedestrians, etc.

![Reward function examples](/output_images/reward_values.png)

*For more information on the derivation of the reward function see Behavioural-Cloning-CV.ipynb*

#### State Approximation
For solving the state discretization problem, I've designed a network loosely based on the LeNet architecture. The input feeds through input preprocessing layers where it crops and normalises the input.  After normalisation, repeating convolutional layers with 64, 32, 16 and 8 filters of 5x5, 3x3 and 2x2 detectors, consecutively, extract meaningful features from the input.  

An average pooling layer downsamples each convolution into rectifiers, for non-linearity, condensing the input into a final average pooling layer. The final pooling layer is responsible for extracting spatial respresentations from each convolution into a compressed output.  Hidden layers of sizes 256 and 128 with parametric rectifiers are computed, along with a dropout rate of 20%, for providing latent features for the final linear layer, where each unit is a steering action.

##### Feature Detector architecture (KaNet)

Layer (type)                     Output Shape          Param #     Connected to
====================================================================================================
input_1 (InputLayer)             (None, 160, 320, 3)   0
____________________________________________________________________________________________________
cropping2d_1 (Cropping2D)        (None, 90, 320, 3)    0           input_1[0][0]
____________________________________________________________________________________________________
lambda_1 (Lambda)                (None, 90, 320, 3)    0           cropping2d_1[0][0]
____________________________________________________________________________________________________
resize_1 (Resize)                (None, 45, 160, 3)    0           lambda_1[0][0]
____________________________________________________________________________________________________
convolution2d_1 (Convolution2D)  (None, 41, 156, 64)   4864        resize_1[0][0]
____________________________________________________________________________________________________
averagepooling2d_1 (AveragePooli (None, 19, 77, 64)    0           convolution2d_1[0][0]
____________________________________________________________________________________________________
activation_1 (Activation)        (None, 19, 77, 64)    0           averagepooling2d_1[0][0]
____________________________________________________________________________________________________
convolution2d_2 (Convolution2D)  (None, 17, 75, 32)    18464       activation_1[0][0]
____________________________________________________________________________________________________
averagepooling2d_2 (AveragePooli (None, 16, 74, 32)    0           convolution2d_2[0][0]
____________________________________________________________________________________________________
activation_2 (Activation)        (None, 16, 74, 32)    0           averagepooling2d_2[0][0]
____________________________________________________________________________________________________
convolution2d_3 (Convolution2D)  (None, 15, 73, 16)    2064        activation_2[0][0]
____________________________________________________________________________________________________
averagepooling2d_3 (AveragePooli (None, 14, 72, 16)    0           convolution2d_3[0][0]
____________________________________________________________________________________________________
activation_3 (Activation)        (None, 14, 72, 16)    0           averagepooling2d_3[0][0]
____________________________________________________________________________________________________
averagepooling2d_4 (AveragePooli (None, 7, 36, 16)     0           activation_3[0][0]
____________________________________________________________________________________________________
flatten_1 (Flatten)              (None, 4032)          0           averagepooling2d_4[0][0]
____________________________________________________________________________________________________
dense_1 (Dense)                  (None, 256)           1032448     flatten_1[0][0]
____________________________________________________________________________________________________
prelu_1 (PReLU)                  (None, 256)           256         dense_1[0][0]
____________________________________________________________________________________________________
dropout_1 (Dropout)              (None, 256)           0           prelu_1[0][0]
____________________________________________________________________________________________________
dense_2 (Dense)                  (None, 128)           32896       dropout_1[0][0]
____________________________________________________________________________________________________
prelu_2 (PReLU)                  (None, 128)           128         dense_2[0][0]
____________________________________________________________________________________________________
dropout_2 (Dropout)              (None, 128)           0           prelu_2[0][0]
____________________________________________________________________________________________________
dense_4 (Dense)                  (None, 13)            1677        dropout_2[0][0]
____________________________________________________________________________________________________
activation_4 (Activation)        (None, 13)            0           dense_4[0][0]
====================================================================================================
Total params: 1,092,797
Trainable params: 1,092,797
Non-trainable params: 0

### Training
The KaNet is then trained using this pre-recorded data using the Adam optimiser with default parameters.  The loss function used in optimisation is the cross entropy loss, trained over 40 epochs with a batch size of 832 and a mini-batch size of 32.  During training a stratified split of samples (~ 2 per class) was used in order to prevent catastrophic forgetting during optimisation.

In order to prevent overfitting, a dropout rate of 20% was utilised in supervised pretraining along with weight decay of 0.0001. 

Samples were augmented with a random mean 10% shift per-sample. In addition a custom callback designed to mitigate overfitting was used during training which stops training if there is signs of overfitting, in this case at epoch 27.

Once training is complete, the softmax output layer is pruned from the network and a Deep Q Network layer - a series of dense rectifier layers with a final linear layer, is then added and trained as part of a Q-Learning model using the standard Q learning update rule.

![Training and validation learning curve](/output_images/training_curve.png)

#### Q-Network
To construct the Q value function approximator, the intermediary output of the KaNet, in this case the final dropout layer just before the softmax layer is used as the input to the Q Network.  Two additional dense layers with rectifiers are appended with a final dense linear layer as output, matching the Q values for each action.

*See appended layers below:*
____________________________________________________________________________________________________
dense_4 (Dense)                  (None, 64)            8256        dropout_2[0][0]
____________________________________________________________________________________________________
activation_3 (Activation)        (None, 64)            0           dense_4[0][0]
____________________________________________________________________________________________________
dense_5 (Dense)                  (None, 32)            2080        activation_3[0][0]
____________________________________________________________________________________________________
activation_4 (Activation)        (None, 32)            0           dense_5[0][0]
____________________________________________________________________________________________________
dense_6 (Dense)                  (None, 13)            429         activation_4[0][0]
____________________________________________________________________________________________________
activation_5 (Activation)        (None, 13)            0           dense_6[0][0]
====================================================================================================

To train the reinforcement learning model, weights were optimised using RMSProp with a learning rate of 0.0001, after seeing a single sample. Online learning was allowed to run for X epochs, where a single epoch was one complete pass around the track.

Training the Deep-Q Network utilises the standard q-learning update rule:
$$ Q(state^{t}, action^{t}) = reward^{t+1} + gamma * Q_{max}(state^{t+1}, action_{argmax}^{t+1}) $$

*Here we update the Q output layer for the previous timestep with the observed reward of the current state, from the action that maximises the value function.*

### Steering Resolution and Smoothing
Generally speaking, an agent needs to have a set of discrete actions within an environment to satisy our MDP requirement.  To do this, the steering angle from each frame has been encoded into a set of actions from -50 to +50 degrees with approximate variance increments.  During prediction time we resolve the step size by applying a weighted smoothing function so the car drives in a nice orderly fashion.

Putting it all together, the KaNet becomes the pretrained feature detector providing the Q network, the state value approximation function it requires to learn an optimal autonomous driving policy.

**Experience Replay:**
After the Q model has been initialised, we run experience replay over a sequence of sequential training data to prime the network.  This is important as after the network has been initialised the weights have not yet converged.  The process of experience replay is taking a selection of random batches from a memory sequence and iterating over it sequentially, then updating the model with the observed reward for the new state after taking [x] action from state [s].  Following this, the Deep Q Network is then primed and ready to undertake online learning.  Experience replay is generally not necessary but helps in this case.

### Reinforcement Learning in Autonomous Mode
During autonomous driving, when online learning is enabled, each frame is passed through the entire DeepQ network which predicts a linear output of [N] actions which provides the action-selection function.  The discounted action which maximises the reward is chosen, then once that action is executed the Q value function at t-1 is updated with the observed reward from the new state.  This temporality of the state-action-reward process allows the network to reinforce its belief of the environment and proper driving behaviour.

*As part of online learning I introduced a new method called "memory sprints" where instead of annealing the epsilon-greedy parameter over time, the model iterates in sprints of 50 epochs, online and offline, consecutively.  This allows the network to gracefully overcome any bad habits it picks up along the way during training...*


### Files in this repo:
 - Behavioural-Cloning.ipynb (Jupyter notebook with reward function derivations)
 - extensions/* (custom keras modules used in model building)
 - callbacks.py (Custom callback to prevent overfitting)
 - cvext.py (low level computer vision functions for tracking road edges)
 - road.py (high level road edge detection module)
 - densenet.py (DenseNet architecture)
 - kanet.py (KaNet architecture)
 - generator.py (module for generating data during training)
 - modelutils.py (module with model training helper functions)
 - model.py (model training script)
 - modelconfig.py (configuration file for persisting model building for saving/loading)
 - qmodel.py (Q Learning model class object)
 - tracker.py (Lane tracking module)
 - video.py (Used to create videos of driving data)

### Running the Code:
To run the driving simulator in autonomous mode, initialise the autonomous driving model by calling drive.py.  Once the model is initialised, start the Udacity driving simulator in autonomous mode and watch the magic happen.  If an existing Q-Learning model is not found a new DeepQNet is constructed with the supplied network weights file by calling `python drive.py <model_def>.h5` etc.

Enabling reinforcement learning is achieved through the 'reinforce' boolean argument, this is enabled by default when a new QLearning model object is created.
 
*To run the simulator in ludicrous mode, pass in True for the 'ludicrous' argument. :)*

#### Issues
The limit of this model is the reward function, it becomes corrupt when the car approaches road edges head on and loses sensitivity.  This needs to be resolved so that the QLearner converges properly.



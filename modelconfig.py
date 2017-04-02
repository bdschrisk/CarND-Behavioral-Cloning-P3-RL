### Model Configuration File ###

### Pretraining config ###
feature_training_dir = "./data/TRAIN/"
feature_training_image_dir = feature_training_dir + "/IMG"
feature_training_file = feature_training_dir + "/driving_log.csv"
feature_training_skip_header = True
feature_training_steering_col = 3
feature_training_image_col = 0
feature_training_image_relative = True

# The model and drive files use this file to construct the model
# model checkpoint path pattern, used for restoring models and saving checkpoints in training
checkpoint_pattern = "./model/checkpoint-*.h5"
# models path
model_path = "./model/"

# returns the class ranges, using -1 to +1 interval
class_ranges = [-1.0, -0.72, -0.36, -0.18, -0.12, -0.06, 0, +0.06, +0.12, +0.18, +0.36, 0.72, +1.0]
img_dim = (160, 320, 3)
resize_factor = 0.5

### Feature Detection Network params ###
layers = 22
growth_rate = 12
dropout = 0.2
noise = 0.3
bottleneck = True
reduction = 0.5

### Optimisation parameters ###
train_split = 0.7

# training params
batch_size = 32
max_epochs = 60
epoch_train_samples = batch_size * len(class_ranges) * 3 # 66.6% training
#epoch_train_samples = len(X_train) # DEFAULT
epoch_test_samples = batch_size * len(class_ranges) # 33.3% test
#epoch_test_samples = len(X_test)

# optimizer params
learning_rate = 0.001
momentum = 0.9
weight_decay = 0.001
use_nesterov = True

### Q Learning config ###
Q_layers = [] #[64, 32] # learning layers of the Q Network
Q_prune = 0 # layers to prune, set to 0 due to keras bug.
Q_minreward = -1 # minimum reward
Q_maxreward = 1 # maximum reward
Q_gamma = 0.9 # gamma param, favour long-term rewards
Q_epsilon = 0.8 # epsilon-greedy parameter, used during sprints
Q_interval = 500 # interval for saving
Q_sprint = 50 # how long each learning sprint is

experience_memory_length = 50
experience_count = 4096 # number of total experiences during experience replay
experience_skip_count = 3 # number of frames to skip during experience replay

online_learning_throttle = 0.04 # May need to adjust this depending on system resources
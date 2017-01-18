
from keras.models import Sequential
from keras.layers import Convolution2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense
from preprocessing.augmentation_policy import augmentation_policy
from deep_networks.lenet_cnn import lenet_cnn
import os
import sys


### SET UP PARAMETERS

# Set default parameters
if len(sys.argv)==1:
    # Get current folder
    current_folder = os.getcwd()
    # Input folder
    input_data_folder = os.path.join(os.path.dirname(current_folder), "data")
    # Input size
    input_size = [150, 150]
else:
    # Input folder will be the first argument
    input_data_folder = sys.argv[1]
    # Input size
    input_size = [sys.argv[2], sys.argv[3]]

# Temporal folder will be inside the data folder
models_folder = os.path.join(input_data_folder, "models")
if not os.path.exists(models_folder):
    os.makedirs(models_folder)



### SET UP THE NETWORK ARCHITECTURE

# Use LeNet with precision and recall metrics
model = lenet_cnn(['accuracy','precision','recall'], input_size)



### SET UP THE TRAINING AND VALIDATION DATA GENERATION POLICIES

# Get a training data generation policy
train_datagen = augmentation_policy("training")
# And a validation data policy too
validation_datagen = augmentation_policy("validation")

# this is a generator that will read pictures found in
# subfolers of 'data/train', and indefinitely generate
# batches of augmented image data
train_generator = train_datagen.flow_from_directory(
        os.path.join(input_data_folder, 'train'),  # this is the target directory
        target_size=(input_size[0], input_size[1]),  # all images will be resized to input_size[0] x input_size[1]
        batch_size=32,
        class_mode='binary')  # since we use binary_crossentropy loss, we need binary labels

# this is a similar generator, for validation data
validation_generator = validation_datagen.flow_from_directory(
        os.path.join(input_data_folder, 'validation'),
        target_size=(input_size[0], input_size[1]),
        batch_size=32,
        class_mode='binary')



### TRAIN THE MODEL

model.fit_generator(
        train_generator,
        samples_per_epoch=630,
        nb_epoch=50,
        validation_data=validation_generator,
        nb_val_samples=270,
        class_weight=[0.82, 0.18])



### SAVE THE WEIGHTS

model.save_weights(os.path.join(models_folder,'lecun_example_try.h5'))  # always save your weights after training or during training

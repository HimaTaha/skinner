
from keras.models import Sequential
from keras.layers import Convolution2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense
from preprocessing.augmentation_policy import augmentation_policy
from deep_networks.lenet_cnn import lenet_cnn
import os


# Input folder
input_data_folder = "/Users/ignaciorlando/Documents/Skinner/data"
# Temporal folder
tmp_folder = "/Users/ignaciorlando/Documents/Skinner/data/tmp"


### SET UP THE NETWORK ARCHITECTURE
model = lenet_cnn()


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
        target_size=(150, 150),  # all images will be resized to 150x150
        batch_size=32,
        class_mode='binary')  # since we use binary_crossentropy loss, we need binary labels

# this is a similar generator, for validation data
validation_generator = validation_datagen.flow_from_directory(
        os.path.join(input_data_folder, 'validation'),
        target_size=(150, 150),
        batch_size=32,
        class_mode='binary')


### TRAIN THE MODEL

model.fit_generator(
        train_generator,
        samples_per_epoch=900,
        nb_epoch=50,
        validation_data=validation_generator,
        nb_val_samples=270)
model.save_weights(os.path.join(tmp_folder,'lecun_example_try.h5'))  # always save your weights after training or during training

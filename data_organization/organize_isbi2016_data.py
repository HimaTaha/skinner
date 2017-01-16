
import os
import hashlib
import zipfile
import csv
import math
import getpass
import sys
import platform

# Set default parameters
if len(sys.argv) < 3:
    # Get current folder
    current_folder = os.getcwd()
    # Set source folder where ISBI2016_ISIC_Part3_Training_Data.zip is
    if platform.system()=='Linux':
        downloaded_dataset_folder = os.path.join("/Users", getpass.getuser(), "Downloads")
    else:
        downloaded_dataset_folder = os.path.join("/home", getpass.getuser(), "Downloads")
    # Destination folder
    destination_folder = os.path.join(os.path.dirname(current_folder), "data")
else:
    # Set source folder where ISBI2016_ISIC_Part3_Training_Data.zip is
    downloaded_dataset_folder = sys.argv[1]
    # and set also the destination folder where the organized data set will be saved
    destination_folder = sys.argv[2]

print(downloaded_dataset_folder)
print(destination_folder)

# Prepare filenames
training_dataset_name = "ISBI2016_ISIC_Part3_Training_Data"
training_dataset_labels_name = "ISBI2016_ISIC_Part3_Training_GroundTruth.csv"

# tmp_folder will save our intermediate results
tmp_folder = os.path.join(destination_folder, "tmp")

# If tmp_folder do not exists, then we will create it
if not os.path.exists(tmp_folder):
    os.makedirs(tmp_folder)

# If the data set is not already unzipped
if not os.path.exists(os.path.join(tmp_folder, training_dataset_name)):
    # Prepare data set filename, including the path to it
    downloaded_dataset_filename = os.path.join(downloaded_dataset_folder, training_dataset_name + ".zip")
    # If the file exists
    if os.path.exists(downloaded_dataset_filename):
        # Unzip that file in the tmp folder
        zip_dataset_ref = zipfile.ZipFile(downloaded_dataset_filename, 'r')
        zip_dataset_ref.extractall(tmp_folder)
        zip_dataset_ref.close()
    else:
        raise FileNotFoundError("Couldn't find " + training_dataset_name + " in " + downloaded_dataset_folder)

# Prepare output folders for training and validation data
training_data_folder = os.path.join(destination_folder, "train")
validation_data_folder = os.path.join(destination_folder, "validation")

# Create training folder if it does not exist
if not os.path.exists(training_data_folder):
    os.makedirs(training_data_folder)
    # with 2 extra folders, one for each label
    os.makedirs(os.path.join(training_data_folder, 'benign'))
    os.makedirs(os.path.join(training_data_folder, 'malignant'))

# Create validation folder if it does not exist
if not os.path.exists(validation_data_folder):
    os.makedirs(validation_data_folder)
    # with 2 extra folders, one for each label
    os.makedirs(os.path.join(validation_data_folder, 'benign'))
    os.makedirs(os.path.join(validation_data_folder, 'malignant'))

# Setup labels path with filename
training_dataset_full_filename = os.path.join(downloaded_dataset_folder, training_dataset_labels_name)

# Check if the file exists
if not os.path.exists(training_dataset_full_filename):

    # If it does not exist, throw an exception
    raise FileNotFoundError("Couldn't find " + training_dataset_labels_name + " in " + downloaded_dataset_folder)

else:

    # Read the CSV file with the labels
    with open(training_dataset_full_filename, newline='\n') as csvfile:
        # Prepare the CSV parser
        my_file_reader = csv.reader(csvfile, delimiter=',')
        # Identify the number of images
        num_images = sum(1 for row in my_file_reader)
        csvfile.seek(0)
        print(str(num_images) + " images in " + training_dataset_name)
        # First 70% images will be used for training and the remaining portion will be used for validation
        num_training_images = math.floor(num_images * 0.7)
        print(str(num_training_images) + " will be used for training, and " + str(num_images - num_training_images) + " for validation")
        # For each row in the csv file...
        images_counter = 1
        print("Copying images...")
        for row in my_file_reader:
            # Get current image filename
            current_image_filename = row[0] + ".jpg"
            # Control if the image has to go to the training or validation set
            if images_counter <= num_training_images:
                # cut and paste each file to the corresponding label folder on the training data
                os.rename(os.path.join(tmp_folder, training_dataset_name, current_image_filename), os.path.join(training_data_folder, row[1], current_image_filename))
            else:
                # cut and paste each file to the corresponding label folder on the validation data
                os.rename(os.path.join(tmp_folder, training_dataset_name, current_image_filename), os.path.join(validation_data_folder, row[1], current_image_filename))
            images_counter = images_counter + 1


# Delete tmp folder
os.rmdir(os.path.join(tmp_folder, training_dataset_name))

# Print a message saying everything was fine
print("Congratulations! Everything was fine :-)")

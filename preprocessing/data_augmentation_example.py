
from keras.preprocessing.image import array_to_img, img_to_array, load_img
from augmentation_policy import augmentation_policy
import os

# Import an augmentation policy
datagen = augmentation_policy("training")

# Load an image just to play around with it
img = load_img('/Users/ignaciorlando/Documents/Skinner/data/train/benign/ISIC_0000000.jpg')  # this is a PIL image
x = img_to_array(img)  # this is a Numpy array with shape (3, 150, 150)
x = x.reshape((1,) + x.shape)  # this is a Numpy array with shape (1, 3, 150, 150)

# Create output folder
output_folder_for_previews = '/Users/ignaciorlando/Documents/Skinner/data/tmp/preview'
if not os.path.exists(output_folder_for_previews):
    os.makedirs(output_folder_for_previews)

# the .flow() command below generates batches of randomly transformed images
# and saves the results to the `preview/` directory
i = 0
for batch in datagen.flow(x, batch_size=1,
                          save_to_dir=output_folder_for_previews, save_prefix='ISIC_0000000', save_format='jpg'):
    i += 1
    if i > 20:
        break  # otherwise the generator would loop indefinitely

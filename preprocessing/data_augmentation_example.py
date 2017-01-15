
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
import os

datagen = ImageDataGenerator(
        # rotation_range: we want any rotation, as dermathological image can be
        # taken at different angles.
        rotation_range=180,
        # width_shift_range & height_shift_range: shifts have to be more convervative as
        # in general our lesions will be pretty centered
        width_shift_range=0.3,
        height_shift_range=0.3,
        # rescale: is used to normalize intensities between 0 and 1
        rescale=1./255,
        # shear_range: is related with deformations, we don't want to deformate
        # our images too much
        shear_range=0.1,
        # zoom_range: we want to zoom in a bit too
        zoom_range=0.3,
        # horizontal_flip: is a pretty good idea too
        horizontal_flip=True,
        fill_mode='nearest')

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

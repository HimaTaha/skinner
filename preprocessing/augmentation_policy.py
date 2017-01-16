
from keras.preprocessing.image import ImageDataGenerator

def augmentation_policy(data_type):

    if (data_type=="training"):
        # datagen will be as follows:
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
                # and fill_mode will be nearest neighbor
                fill_mode='nearest')
    else:
        # datagen will be only rescale
        datagen = ImageDataGenerator(
                rescale=1./255)

    # return our datagen object
    return datagen

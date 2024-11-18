import pandas as pd
import numpy as np
import tensorflow as tf
import glob
import os
from keras import preprocessing
import config
import kaggle

# Assign the location path of the dataset
data_path = config.data_path

# Set the dataset paths
train_path = config.train_path
test_path = config.test_path
pred_path = config.pred_path

# Determine frequently used constants
image_size = config.image_size
img_height = config.img_height
img_width = config.img_width
img_channels = config.img_channels
batch_size = config.batch_size
class_names = config.class_names

# Process the images into TF datasets
(train_set, validation_set) = preprocessing.image_dataset_from_directory(
    train_path,
    labels = 'inferred',
    label_mode = 'int',
    class_names = class_names,
    color_mode = 'rgb',
    batch_size = batch_size,
    image_size = image_size,
    shuffle = True,
    seed = 1234,
    validation_split = 0.2,
    subset = 'both',
)

test_set = preprocessing.image_dataset_from_directory(
    test_path,
    labels = 'inferred',
    label_mode = 'int',
    class_names = class_names,
    color_mode = 'rgb',
    batch_size = batch_size,
    image_size = image_size,
    shuffle = True,
    seed = 1234,
)

# Normalize image data
def process(image, label):
    """
    Normalizes image data and casts it to float32.

    Parameters:
        image (Tensor): Input image tensor with pixel values in the range [0, 255].
        label (Tensor): Corresponding label tensor for the image.

    Returns:
        image, label (Tuple): Normalized image tensor with pixel values in the range [0, 1], and the corresponding label tensor.
    """
    image = tf.cast(image / 255. ,tf.float32)
    return image, label

train_set = train_set.map(process)
validation_set = validation_set.map(process)
test_set = test_set.map(process)

# Save the datasets
train_set.save('data/processed/train_set')
validation_set.save('data/processed/validation_set')
test_set.save('data/processed/test_set')
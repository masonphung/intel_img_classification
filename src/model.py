# Load libraries
import pandas as pd
import numpy as np
import glob
import os
import config

from sklearn.metrics import confusion_matrix
import tensorflow as tf
from tensorflow import keras
import keras_tuner as kt
from keras import preprocessing
from tensorflow.keras.layers import Input, Dense, Flatten, Dropout, Conv2D, MaxPooling2D, RandomFlip, RandomRotation, RandomZoom

# Image data properties
image_size = config.image_size
img_height = config.img_height
img_width = config.img_width
img_channels = config.img_channels
batch_size = config.batch_size
class_names = config.class_names

# Define `build_model` to build the deep NN model
def build_model(hp):
    """
    Returns a compiled keras neural network model with a search space initialized with
    to-be-tuned hyperparameters.
    
    Parameters:
        hp (HyperParameters class): Define the hyperparameters when building the model
    
    Returns:
        model (keras.models.Sequential): The constructed & compiled Keras model.
    """
    model = keras.Sequential()
    # Add the Input layer
    model.add(Input(shape = (img_height, img_width, img_channels)))
    # Add a data augmentation layer
    model.add([
        RandomFlip('horizontal'),
        RandomRotation(0.1),
        RandomZoom(0.2)
    ])
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                        
    # Tuning the number of layers
    num_batches = hp.Int('num_batches', min_value=1, max_value=5, step=1)        
    for i in range(num_batches):
        num_layers = hp.Int(f'num_layers_{i+1}', min_value=1, max_value=3, step=1)
        for _ in range(num_layers):
            # Conv2D layers
            model.add(Conv2D(
                filters = hp.Int(f'conv_filter_{i+1}', min_value=32, max_value=256, step=32*(i+1)),
                kernel_size = hp.Choice('conv_kernel', values = [3,5]),
                activation = 'relu',
                padding = 'same'
            ))
            # MaxPooling2D layers
            model.add(MaxPooling2D(
                pool_size = (2,2),
                padding = 'same'
            ))    
        # Tune a dropout layer
        dropout_rate = hp.Float(f"dropout_rate{i+1}", min_value = 0.1, max_value = 0.5, step = 0.1)
        # Add a dropout layer
        model.add(Dropout(rate = dropout_rate))
        
    # Add a Flatten layer
    model.add(Flatten())
    # Add a Dense model layer
    model.add(Dense(
        units = hp.Int('units', min_value=32, max_value=512, step=32),
        activation = 'relu'
    ))
    # Add a dropout layer
    model.add(Dropout(rate=hp.Float('final_dropout_rate', min_value=0.1, max_value=0.5, step=0.1)))
    # Add the output layer
    model.add(Dense(units = len(class_names), activation = 'softmax'))
    
    # Tune the learning rate
    learning_rate = hp.Float("lr", min_value=1e-4, max_value=1e-1, sampling="log")
    # Configure the model to fit the case 
    model.compile(
        optimizer = keras.optimizers.Adam(learning_rate = learning_rate),
        loss = keras.losses.SparseCategoricalCrossentropy(from_logits=False),
        metrics = ['accuracy']
    )
    return model
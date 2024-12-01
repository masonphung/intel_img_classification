# Load libraries
import pandas as pd
import numpy as np
import glob
import os
from src import config

from sklearn.metrics import confusion_matrix
import tensorflow as tf
from tensorflow import keras
import keras_tuner as kt
from keras import preprocessing
from tensorflow.keras.layers import Input, Dense, Flatten, Dropout, Conv2D, MaxPooling2D
from tensorflow.keras.layers import RandomFlip, RandomRotation, RandomZoom

# Image data properties
image_size = config.image_size
img_height = config.img_height
img_width = config.img_width
img_channels = config.img_channels
batch_size = config.batch_size
class_names = config.class_names

# Define `build_model` to build the deep NN model
def build_model():
    """
    Returns a compiled keras neural network model
    
    Returns:
        model (keras.models.Sequential): The constructed & compiled Keras model.
    """
    # Build model architecture
    model = keras.models.Sequential([
        Input(shape=(img_height, img_width, img_channels)),
        # Data augmentation layer
        RandomFlip('horizontal'),
        RandomRotation(0.2),
        RandomZoom(0.3),
        
        Conv2D(filters=32, kernel_size=(3, 3), activation='relu', padding='same'),
        MaxPooling2D(pool_size=(2, 2), padding = 'same'),
        Conv2D(filters=32, kernel_size=(3, 3), activation='relu', padding='same'),
        MaxPooling2D(pool_size=(2, 2), padding = 'same'),
        Dropout(rate=0.2),
        
        Conv2D(filters=64, kernel_size=(3, 3), activation='relu', padding='same'),
        MaxPooling2D(pool_size=(2, 2), padding = 'same'),
        Conv2D(filters=64, kernel_size=(3, 3), activation='relu', padding='same'),
        MaxPooling2D(pool_size=(2, 2), padding = 'same'),
        Dropout(rate=0.3),
        
        Conv2D(filters=128, kernel_size=(3, 3), activation='relu', padding='same'),
        MaxPooling2D(pool_size=(2, 2), padding = 'same'),
        Conv2D(filters=128, kernel_size=(3, 3), activation='relu', padding='same'),
        MaxPooling2D(pool_size=(2, 2), padding = 'same'),
        Dropout(rate=0.4),
        
        Conv2D(filters=256, kernel_size=(3, 3), activation='relu', padding='same'),
        MaxPooling2D(pool_size=(2, 2), padding = 'same'),
        Conv2D(filters=256, kernel_size=(3, 3), activation='relu', padding='same'),
        MaxPooling2D(pool_size=(2, 2), padding = 'same'),
        Dropout(rate=0.5),
        
        # Flatten and Dense layers
        Flatten(),
        Dense(units=512, activation='relu'),
        Dropout(rate=0.5),
        Dense(units=len(class_names), activation='softmax')
    ])


    # Compile with suitable parameters
    model.compile(
        optimizer = keras.optimizers.Adam(learning_rate = 0.001),
        loss = keras.losses.SparseCategoricalCrossentropy(from_logits=False),
        metrics = ['accuracy']
    )
    return model
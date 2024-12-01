# Load libraries
from src import config
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Input, Dense, RandomFlip, RandomRotation, RandomZoom
from tensorflow.keras.applications import ResNet50V2
import ssl
ssl._create_default_https_context = ssl._create_unverified_context

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
    # Create a pre-trained mobilenet model instance
    mobilenet_model = ResNet50V2(
        weights = "imagenet",
        input_shape = (img_height, img_width, img_channels),
        include_top = False,
        pooling = "avg",
        name = "resnet50"
    )

    for layer in mobilenet_model.layers:
        layer.trainable = False

    mobile_model = keras.models.Sequential([
        # Input layer
        Input(shape = (img_height, img_width, img_channels)),
        # Data augmentation
        RandomFlip('horizontal'),
        RandomRotation(0.1),
        RandomZoom(0.2),
        # MobileNet model
        mobilenet_model,
        # Output layer
        Dense(len(class_names), activation = 'softmax')
    ])

    mobile_model.compile(
        optimizer = keras.optimizers.Adam(learning_rate = 0.001), 
        loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits = False), 
        metrics = ['accuracy']
    )
    return mobile_model


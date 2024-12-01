import config
from tensorflow import keras
from src.cnn_model import build_model
import keras_tuner as kt
from tensorflow.keras.layers import Input, Dense, Flatten, Dropout, Conv2D, MaxPooling2D, RandomFlip, RandomRotation, RandomZoom
from src import utils

early_stopping = keras.callbacks.EarlyStopping(monitor = 'val_loss', min_delta = 0, patience = 3)
train_set, validation_set, test_set = utils.load_processed_data()

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
    model.add(RandomFlip('horizontal')),
    model.add(RandomRotation(0.1)),
    model.add(RandomZoom(0.2)),
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                        
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
        optimizer = keras.optimizers.legacy.Adam(learning_rate = learning_rate),
        loss = keras.losses.SparseCategoricalCrossentropy(from_logits=False),
        metrics = ['accuracy']
    )
    return model

# Tune model
def tune_model():
    """
    This function tunes the hyperparameters of a convolutional neural network (CNN) model using Bayesian Optimization.
    It utilizes the Keras Tuner library to search for the best hyperparameters that maximize the validation accuracy.

    Parameters:
    None

    Returns:
    A hyperparameter tuned Keras CNN model.
    """
    # Set up the tuner class to search
    tuner = kt.BayesianOptimization(
        hypermodel = build_model,
        objective = kt.Objective('val_accuracy', 'max'),
        max_trials = 5,
        num_initial_points = 2,
        overwrite = True,
        project_name = 'cnn_fine_tuned'
    )

    # Take a look at the search space
    tuner.search_space_summary()
    
    # Search for the best hyperparameters
    tuner.search(
        train_set,
        validation_data = validation_set,
        epochs = 25,
        verbose = 1,
        callbacks = [early_stopping]
        )

    # Print the best hyperparameters and model (Top 1)
    topN = 1
    for x in range(topN):
        best_hp = tuner.get_best_hyperparameters(topN)[x]
        print(best_hp.values)
        print(tuner.get_best_models(topN)[x].summary())
    
    optimal_model = build_model(best_hp)
    optimal_model.save('models/fine_tuned_cnn.h5')
    print('Model saved!')
    return  optimal_model, best_hp

if __name__ == "__main__":
    best_model, best_hps = tune_model()
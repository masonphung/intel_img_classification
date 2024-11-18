import tensorflow as tf
from tensorflow import keras
from model import build_model
import keras_tuner as kt

early_stopping = keras.callbacks.EarlyStopping(monitor = 'val_loss', min_delta = 0, patience = 3)

# Load dataset
train_set = tf.data.Dataset.load('data/processed/train_set')
validation_set = tf.data.Dataset.load('data/processed/validation_set')
test_set = tf.data.Dataset.load('data/processed/test_set')

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
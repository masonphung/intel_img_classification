from tensorflow import keras
import tensorflow as tf
from src import cnn_model, mobilenet_model,resnet50_model, vgg19_model, utils
import os

# Suppress TensorFlow logs
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
tf.get_logger().setLevel('ERROR')

# Define available models
MODELS = {
    1: ('cnn_model', cnn_model.build_model),
    2: ('mobilenet_model', mobilenet_model.build_model),
    3: ('vgg19_model', vgg19_model.build_model),
    4: ('resnet50_model', resnet50_model.build_model)
}

def train_model(model_name, build_model_fn):
    # Build the selected model
    model = build_model_fn()

    # Load datasets
    train_set = tf.data.Dataset.load('data/processed/train_set')
    validation_set = tf.data.Dataset.load('data/processed/validation_set')

    # Set callbacks: higher patience for the self-built model, add rlronp for pre-trained models
    if model_name=='cnn_model':
        early_stopping = keras.callbacks.EarlyStopping(
            monitor='val_loss',
            min_delta=0,
            patience=8,
        )
        # Train th
        result = model.fit(
            train_set,
            validation_data=validation_set,
            epochs=50,
            callbacks=[early_stopping]
        )
    else:
        early_stopping = keras.callbacks.EarlyStopping(
            monitor='val_loss',
            min_delta=0,
            patience=4,
        )

        rlronp = tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", 
            factor=0.4,
            patience=2, 
            verbose=1, 
            mode="auto")
        
        # Train with early stop and dynamic LR
        result = model.fit(
            train_set,
            validation_data=validation_set,
            epochs=50,
            callbacks=[early_stopping, rlronp]
        )

    # Save the model with a name based on the selected model
    save_path = f'models/{model_name}.h5'
    model.save(save_path)
    print(f'Model saved at {save_path}!')

    # Plot the accuracy and loss of the model
    print("The accuracy plot of the model:\n")
    utils.plot_accuracy_loss(result)


if __name__ == "__main__":
    # Display available models
    print("Select a model to train:")
    for key, (model_name, _) in MODELS.items():
        print(f"{key}: {model_name}")

    # Get user input
    try:
        choice = int(input("\nEnter the number corresponding to the model: ").strip())
        if choice not in MODELS:
            raise ValueError(f"Invalid choice: {choice}")

        # Get the selected model's name and function
        selected_model_name, build_model_fn = MODELS[choice]
        print(f'{selected_model_name}')
        # Train the selected model
        train_model(selected_model_name, build_model_fn)
        print("Model trained successfully!")

    except ValueError as e:
        print(f"Error: {e}. Please enter a valid number.")
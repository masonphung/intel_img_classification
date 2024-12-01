import tensorflow as tf
from tensorflow.keras.models import load_model
from sklearn.metrics import confusion_matrix
from src import utils
import os

# Suppress TensorFlow logs
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # 0 = all logs, 1 = info, 2 = warnings, 3 = errors
# Optionally reset TensorFlow's logger if needed
tf.get_logger().setLevel('ERROR')

def test_model(model_name):
    """
    Tests a trained model on the test dataset.

    Parameters:
        model_name (str): Name of the trained model file (e.g., 'cnn_model').

    Returns:
        None: Displays test accuracy, confusion matrix, and misclassified images.
    """
    # Load test set, trained model and training results
    test_set = tf.data.Dataset.load('data/processed/test_set')
    model = load_model(f'models/{model_name}.h5')
    print(f'Model {model_name} loaded!')

    # Predict with the test set
    print('Evaluating the test set...')
    test_loss, test_acc = model.evaluate(test_set)
    print('Test accuracy of the model:', test_acc)

    # Get the predicted, true labels and the test images
    y_true, y_pred = utils.extract_labels(test_set, model)
    test_images = utils.extract_pred_img(test_set)

    # Calculate the confusion matrix
    pred_matrix = confusion_matrix(y_true, y_pred)

    # We'll evaluate model performance by displaying:
    ## 1. The confusion matrix
    utils.plot_conf_matrix(pred_matrix, normalize=True)
    ## 2. Some mislabeled images
    utils.print_most_misclass(pred_matrix, num_of_img=5, y_true=y_true, y_pred=y_pred, test_images=test_images)

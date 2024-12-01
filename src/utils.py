import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from src import config
import glob
from random import randint
from tensorflow.keras.models import load_model
import tensorflow as tf
from sklearn.metrics import classification_report, accuracy_score
from itertools import cycle
from random import choice


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

def display_images(num_images, target_classes, train_path=train_path):
    """
    Display training images from the specified classes with configurable number and layout.
    
    Parameters:
        num_images (int): Total number of images to display.
        target_classes (str or list): Specific class name(s) or 'all' to display all classes.
        train_path (str): Path to the training data directory containing class folders.
        
    Returns:
        None: Displays the images in a grid.
    """
    if target_classes == 'all':
        target_classes = class_names
    elif isinstance(target_classes, str):
        target_classes = [target_classes]
    
    # Validate target classes
    target_classes = [cls for cls in target_classes if cls in class_names]

    # Ensure there are valid classes
    if not target_classes:
        print("No valid classes found.")
        return

    # Create a dictionary of image paths for each class
    class_image_paths = {cls: glob.glob(f'{train_path}/{cls}/*') for cls in target_classes}

    # Remove classes with no images
    class_image_paths = {cls: paths for cls, paths in class_image_paths.items() if paths}

    if not class_image_paths:
        print("No images found for the specified classes.")
        return

    # Cycle through the classes to select images
    image_cycle = cycle(class_image_paths.keys())
    selected_images = []
    for _ in range(num_images):
        cls = next(image_cycle)
        if class_image_paths[cls]:
            selected_images.append((cls, choice(class_image_paths[cls])))

    # Calculate the number of columns and rows
    cols = min(num_images, 6)  # Maximum 6 images per row
    rows = (num_images + cols - 1) // cols  # Dynamic rows based on num_images and cols

    # Create the grid
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 4, rows * 3))
    axes = np.atleast_2d(axes)  # Ensure axes is always 2D for consistent indexing
    axes = axes.flatten()  # Flatten to handle images dynamically

    # Display images
    for idx, (cls, img_path) in enumerate(selected_images):
        img = plt.imread(img_path)
        img_dim = np.array(img)
        axes[idx].imshow(img)
        axes[idx].set_title(f'{cls} {img_dim.shape}')
        axes[idx].axis('off')

    # Hide unused axes
    for ax in axes[len(selected_images):]:
        ax.axis('off')
        
    
def plot_accuracy_loss(history):
    """
    Plot the accuracy and the loss during the training of the neural network
    
    Parameters:
        history: A History object returned by the fit method of a Keras model. It contains the training and validation loss and accuracy for each epoch.
    
    Returns:
        A plt plot presents the training and validation loss and accuracy.
    """
    plt.figure(figsize=(12, 16))

    plt.subplot(4, 2, 1)
    plt.plot(history.history['loss'], label='Loss')
    plt.plot(history.history['val_loss'], label='val_Loss')
    plt.title('Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.xticks(range(len(history.history['loss'])))

    plt.subplot(4, 2, 2)
    plt.plot(history.history['accuracy'], label='accuracy')
    plt.plot(history.history['val_accuracy'], label='val_accuracy')
    plt.title('Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.xticks(range(len(history.history['accuracy'])))

    return plt.show()

def plot_conf_matrix(matrix, normalize=False):
    """
    Visualize a confusion matrix using a heatmap to display the performance of the classification model.
    
    Parameters:
        matrix (np.array): The confusion matrix, where rows represent true labels and columns represent predicted labels.
        normalize (bool): If True, normalizes the matrix to show percentages.
    
    Returns:
        A confusion matrix heat map.
    """
    # Normalize the matrix to show percentages
    if normalize:
        matrix = matrix.astype('float') / matrix.sum(axis=1, keepdims=True)
    
    class_names = config.class_names
    sns.heatmap(
        matrix,
        annot=True,
        fmt='.2%' if normalize else 'd',
        center=0,
        square=True,
        cmap="RdBu_r",
        xticklabels=class_names,
        yticklabels=class_names,
    )

    plt.title('Confusion Matrix between Predicted and True Labels')
    plt.xlabel('Predicted Labels')
    plt.tick_params(axis='x', labelsize=8)
    plt.ylabel('True Labels')
    plt.tick_params(axis='y', labelsize=8) 

    return plt.show()

def extract_labels(dataset, model):
    """
    Extracts true labels, predicted labels from the test dataset using the given model.

    Parameters:
        dataset (tf.data.Dataset): The dataset containing images and their true labels.
        model (tf.keras.Model): The trained CNN model used for predictions.

    Returns:
        tuple: A tuple containing:
            - y_true (list): True labels of the test images.
            - y_pred (list): Predicted labels of the test images.
    """
    y_true = []
    y_pred = []

    # Extract images and true labels from the test set, then get predicted labels
    for images, labels in dataset:
        y_true.extend(labels.numpy())
        preds = model.predict(images, verbose=0)
        y_pred.extend(np.argmax(preds, axis=-1))
    return y_true, y_pred, 

def extract_pred_img(test_set):
    """
    Extracts test images from the test dataset using the given model.

    Parameters:
        test_set (tf.data.Dataset): The test dataset containing images and their true labels.
        model (tf.keras.Model): The trained CNN model used for predictions.

    Returns:
        tuple: A tuple containing:
            - test_images (list): Numpy array representations of the test images.
    """
    test_images = []

    # Extract images and true labels from the test set, then get predicted labels
    for images, _ in test_set:  # Ignore the labels
        test_images.extend(images.numpy())
    return test_images

def class_report(model_name):
    """
    Calculate the accuracy score of the classification model on the both the train and test set.
    Then create a classification report.
    
    Parameters:
        model_name (str): Name of the classification model
    
    Returns:
        Accuracy score and classification report of the model on both training and test set.
    """
    test_set = tf.data.Dataset.load('data/processed/test_set')
    train_set = tf.data.Dataset.load('data/processed/train_set')
    model = load_model(f'models/{model_name}.h5')
    y_true, y_pred = extract_labels(test_set, model)
    y_true_train, y_pred_train = extract_labels(train_set, model)
    
    # Calculate accuracy score
    accuracy = accuracy_score(y_true, y_pred)
    accuracy_train = accuracy_score(y_true_train, y_pred_train)
    
    # Print results
    print("Training Set Accuracy = {:.5f}".format(accuracy_train))
    print("Test Set Accuracy = {:.5f}".format(accuracy), '\n')

    print("Training Set Classification Report:")
    print(classification_report(y_true_train, y_pred_train, digits = 5))
    
    print("Test Set Classification Report:")
    print(classification_report(y_true, y_pred, digits = 5))


def print_most_misclass(matrix, num_of_img, y_true, y_pred, test_images, class_names=config.class_names):
    """
    Identify and display the most misclassified label pair from a confusion matrix, 
    along with the corresponding mislabeled images, their true and predicted labels.

    Parameters:
        matrix (np.array): Confusion matrix showing the counts of true vs. predicted labels.
        num_of_img (int): Number of mislabeled images to display from the most misclassified pair.
        y_true (list): List of true labels for the test set.
        y_pred (list): List of predicted labels for the test set.
        test_images (list): List of test images corresponding to the labels.
        class_names (list): List of class names (default is `config.class_names`).

    Returns:
        None: The function prints out the most misclassified label pair and displays the 
              mislabeled images with their true and predicted labels.
    """
    # Set diagonal to 0 to ignore correct predictions
    conf_matrix_no_diag = matrix.copy()
    np.fill_diagonal(conf_matrix_no_diag, 0)

    # Find the indices of the most misclassified label pair
    most_misclassified = np.unravel_index(np.argmax(conf_matrix_no_diag), conf_matrix_no_diag.shape)
    true_label_idx, pred_label_idx = most_misclassified

    print(f"Most misclassified pair: Misclassified True Label: {class_names[true_label_idx]} as Predicted Label: {class_names[pred_label_idx]}")

    # Filter misclassified images
    mislabeled_indices = np.where(
        (np.array(y_true) == true_label_idx) & (np.array(y_pred) == pred_label_idx)
    )[0]

    # Extract the mislabeled images, predicted labels, and true labels
    mislabeled_images = np.array(test_images)[mislabeled_indices]
    mislabeled_predicted_labels = np.array(y_pred)[mislabeled_indices]
    mislabeled_true_labels = np.array(y_true)[mislabeled_indices]

    # Show the number of total mislabeled images in the most misclassified pair
    print(f"Number of mislabeled images for {class_names[true_label_idx]} predicted as {class_names[pred_label_idx]}: {len(mislabeled_indices)}")

    # Plot the first 5 mislabeled images
    plt.figure(figsize=(15, 4))
    for idx, label in enumerate(mislabeled_predicted_labels[:num_of_img]):
        plt.subplot(1, num_of_img, idx + 1)
        plt.imshow(mislabeled_images[idx])
        plt.xticks([])
        plt.yticks([])
        # Add the true label and predicted label to the title
        true_label_name = class_names[mislabeled_true_labels[idx]]
        predicted_label_name = class_names[label]
        plt.title(f'True: {true_label_name}\nPredicted: {predicted_label_name}')
    plt.suptitle(f'Mislabeled images for {class_names[true_label_idx]} predicted as {class_names[pred_label_idx]}', 
                size=16, weight='bold')
    plt.show()
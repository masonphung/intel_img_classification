import matplotlib as plt

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
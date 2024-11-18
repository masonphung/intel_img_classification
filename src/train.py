import tensorflow as tf
from tensorflow.keras.models import load_model

model = load_model("models/fine_tuned_cnn.h5")
print("Model loaded!")
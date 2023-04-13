import tensorflow as tf
from config import IMG_HEIGHT, IMG_WIDTH, N_IMAGE_CHANNEL

def get_model():
    # create a sequential model
    model = tf.keras.models.Sequential([
        tf.keras.layers.Flatten(input_shape=(IMG_HEIGHT, IMG_WIDTH, N_IMAGE_CHANNEL)),
        tf.keras.layers.Dense(units=1, activation='sigmoid'),
    ])

    return model
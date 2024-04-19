"""
Re-usable functions for predefined models.
"""

import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras import Sequential
import tensorflow_addons as tfa
from tensorflow.keras.applications import resnet50
import tensorflow as tf
import keras_tuner as kt
from tensorflow import keras
from kerastuner.tuners import RandomSearch
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Dropout, GlobalAveragePooling2D, \
    BatchNormalization

# from cnn_utils import get_activation_fn, learning_rate

_seed = 12345
tf.random.set_seed(_seed)

def get_activation_fn(no_of_classses: int):
    if no_of_classses == 2:
        return "sigmoid"
    else:
        return "softmax"

def baseline_model(input_shape, no_of_classes, metrics):
    """
    This function creates a simple CNN model for baseline comparison
    """
    model = Sequential()
    model.add(Conv2D(32, kernel_size=(3, 3), activation="relu", input_shape=input_shape))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(BatchNormalization())

    model.add(Conv2D(64, kernel_size=(3, 3), activation="relu"))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(128, kernel_size=(3, 3), activation="relu"))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(256, kernel_size=(3, 3), activation="relu"))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(512, kernel_size=(3, 3), activation="relu"))
    model.add(BatchNormalization())

    model.add(Flatten())
    model.add(Dense(512, activation="relu"))
    model.add(BatchNormalization())
    model.add(Dense(no_of_classes, activation=get_activation_fn(no_of_classes)))
    opt = tf.optimizers.Adam(learning_rate=1e-4)
    model.compile(optimizer=opt, loss=tf.keras.losses.SparseCategoricalCrossentropy(), metrics=metrics)
    return model


"""
freezing/unfreezing layers in a pretrained model
"""
def freeze_model(model):
    for layer in model.layers:
        layer.trainable = False
    return model


def unfreeze(model, count_of_layers):
    for layer in model.layers[:-count_of_layers]:
        if not isinstance(layer, BatchNormalization):
            layer.trainable = True
    return model

def pretrained_model(model_name, input_shape, no_of_classes, metrics, no_of_layers):
    """
    Helper function for using different pretrained models.
    i used some of the code from here: https://keras.io/examples/vision/image_classification_efficientnet_fine_tuning/
    """
    base_model = None
    if model_name == "ResNet50":
        base_model = tf.keras.applications.resnet50.ResNet50(weights='imagenet', include_top=False, input_shape=input_shape)
        base_model = freeze_model(base_model)
        base_model = unfreeze(base_model, no_of_layers)
    elif model_name == "VGG16":
        base_model = tf.keras.applications.vgg19.VGG19(weights='imagenet', include_top=False, input_shape=input_shape)
        base_model = freeze_model(base_model)
        base_model = unfreeze(base_model, no_of_layers)
    else:
        pass

    model = Sequential()
    model.add(base_model)
    model.add(Flatten())
    model.add(Dense(no_of_classes, activation=get_activation_fn(no_of_classes)))
    opt = tf.optimizers.Adam(learning_rate=2e-5)
    model.compile(optimizer=opt, loss=tf.keras.losses.SparseCategoricalCrossentropy(), metrics=metrics)
    return model

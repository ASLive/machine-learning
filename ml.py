# TensorFlow and tf.keras
import tensorflow as tf
import keras
from keras.models import model_from_json
from keras.layers import Dense, Activation, Flatten, Dropout, Conv2D, MaxPooling2D
from os.path import isfile
from settings import JSON_PATH, WEIGHTS_PATH

def setup():
    num_clusters = 24
    return keras.Sequential([
        Flatten(input_shape=(21, 3)),
        Dense(42069, activation=tf.nn.relu),
        Dense(num_clusters, activation=tf.nn.softmax),
    ])

def compile(model):
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'],)

def ml_model(train_images, train_labels, retrain=False, save=True):
    if (not retrain) and isfile(JSON_PATH) and isfile(WEIGHTS_PATH):
        return read_model()
    else:
        return make_model(train_images, train_labels, save=save)

def make_model(train_images, train_labels, save=True):
    model = setup()
    compile(model)
    model.fit(train_images, train_labels, epochs=69) # train
    return save_model(model) if save else model

def save_model(model):
    # serialize model to json
    model_json = model.to_json()
    with open(JSON_PATH, "w") as json_file:
        json_file.write(model_json)
    # serialize weights to hdf5
    model.save_weights(WEIGHTS_PATH)
    return model

def read_model():
    # load json and create model
    json_file = open(JSON_PATH, 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)
    # load weights into new model
    loaded_model.load_weights(WEIGHTS_PATH)
    compile(loaded_model)
    return loaded_model

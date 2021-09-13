# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# GOOD ONE


import numpy as np
import argparse
import os
import glob
import json

import matplotlib.pyplot as plt

# import keras
# from keras.models import Sequential, model_from_json
# from keras.layers import Dense
# from keras.optimizers import RMSprop
# from keras.callbacks import Callback

from tensorflow import keras
from tensorflow.keras.models import Sequential, model_from_json
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.callbacks import Callback

import tensorflow as tf

from azureml.core import Run
from utils import load_data, one_hot_encode
from sklearn.metrics import precision_score



# Load the training parameters from the parameters file
with open("parameters.json") as f:
    pars = json.load(f)
try:
    train_args = pars["training"]
except KeyError:
    print("Could not load training values from file")
    train_args = {}


n_inputs = train_args["n_inputs"]
n_h1 = train_args["n_h1"]
n_h2 = train_args["n_h2"]
n_outputs = train_args["n_outputs"]
n_epochs = train_args["n_epochs"]
batch_size = train_args["batch_size"]
learning_rate = train_args["learning_rate"]

# parser = argparse.ArgumentParser()
# parser.add_argument('--data-folder', type=str, dest='data_folder', default='data', help='data folder mounting point')
# args = parser.parse_args()

# data_folder = args.data_folder

def preprocess_df(data_folder):
    X_train_path = glob.glob(os.path.join(data_folder, '**/train-images-idx3-ubyte.gz'), recursive=True)[0]
    X_test_path = glob.glob(os.path.join(data_folder, '**/t10k-images-idx3-ubyte.gz'), recursive=True)[0]
    y_train_path = glob.glob(os.path.join(data_folder, '**/train-labels-idx1-ubyte.gz'), recursive=True)[0]
    y_test_path = glob.glob(os.path.join(data_folder, '**/t10k-labels-idx1-ubyte.gz'), recursive=True)[0]

    X_train = load_data(X_train_path, False) / 255.0
    X_test = load_data(X_test_path, False) / 255.0
    y_train = load_data(y_train_path, True).reshape(-1)
    y_test = load_data(y_test_path, True).reshape(-1)

    training_set_size = X_train.shape[0]

    y_train = one_hot_encode(y_train, n_outputs)
    y_test = one_hot_encode(y_test, n_outputs)
    
    data = {"train": {"X": X_train, "y": y_train},
            "test": {"X": X_test, "y": y_test}} 

    return data



def train_model(data): 



    # Build a simple MLP model
    model = Sequential()
    # first hidden layer
    model.add(Dense(n_h1, activation='relu', input_shape=(n_inputs,)))
    # second hidden layer
    model.add(Dense(n_h2, activation='relu'))
    # output layer
    model.add(Dense(n_outputs, activation='softmax'))

    model.summary()

    model.compile(loss='categorical_crossentropy',
                optimizer=RMSprop(lr=learning_rate),
                metrics=['accuracy'])

    model.fit(data["train"]["X"], data["train"]["y"],
                    batch_size=batch_size,
                    epochs=n_epochs,
                    verbose=2,
                    validation_data=(data["test"]["X"], data["test"]["y"]))

    return model


def get_model_metrics(model,data):
    loss, accuracy = model.evaluate(data["test"]["X"], data["test"]["y"], verbose=0)
    metrics = {"accuracy": accuracy}
    return metrics


def main():
    print("Running train.py")


    data_folder = os.path.join(os.path.dirname(os.path.dirname(os.getcwd())), 'data')

    data = preprocess_df(data_folder)

    # Train the model
    model = train_model(data)

    # Log the metrics for the model
    metrics = get_model_metrics(model, data)
    for (k, v) in metrics.items():
        print(f"{k}: {v}")


if __name__ == '__main__':
    main()



import pandas as pd
import numpy as np
import tensorflow as tf

from Preprocessing import prepare_data

from tensorflow.python.keras import models
from tensorflow.python.keras.layers import Dense
from tensorflow.python.keras.layers import Dropout



# takes a neural network's output and converts it into a probability distribution
# where each value is between 0 and 1 and the values add up to 1
last_layer_activation = 'softmax'

validation_percent = 0.2


def mlp(num_layers, num_units, dropout_rate, input_shape, num_classes):
    model = models.Sequential() # create model
    model.add(Dropout(rate=dropout_rate, input_shape=input_shape)) #set shape and dropout rate

    for _ in range(num_layers-1): #add layers
        model.add(Dense(units=num_units, activation='relu'))
        model.add(Dropout(rate=dropout_rate))

    model.add(Dense(units=num_classes, activation=last_layer_activation)) #add last layer
    model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

    return model


def train_model(train_X,
              val_X,
              train_labels,
              val_labels,
              learning_rate=1e-3,
              epochs=1000,
              batch_size=128,
              num_layers=2,
              num_units=64,
              dropout_rate=0.2):


    model = mlp(num_layers, num_units, dropout_rate, train_X.shape[1:], num_classes)

    # Create callback for early stopping on validation loss. If the loss does
    # not decrease in two consecutive tries, stop training.
    callbacks = [tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=2)]

    history = model.fit(
            train_X,
            train_labels,
            epochs=epochs,
            callbacks=callbacks,
            validation_data=(val_X, val_labels),
            verbose=10,  # Logs once per epoch.
            batch_size=batch_size)

    history = history.history
    print('Validation accuracy: {acc}, loss: {loss}'.format(
            acc=history['val_accuracy'][-1], loss=history['val_loss'][-1]))
    end_vals = {key:val[-1] for (key, val) in history.items()}

    return model, end_vals


train_X, val_X, train_labels, val_labels, num_classes, topic_map = prepare_data()
# model, end_vals = train_model( train_X, val_X, train_labels, val_labels)
# print(end_vals)

def test_hyperparameters(num_layers_range=[1, 2, 3],
                        num_units_range=[8, 16, 32, 64],
                        dropout_rate_range=[0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5] ):
    parameter_loss = {}
    for num_layers in num_layers_range:
        for num_units in num_units_range:
            for dropout_rate in dropout_rate_range:
                model, end_vals = train_model( train_X, val_X, train_labels, val_labels
                                   , num_layers = num_layers
                                   , num_units = num_units
                                   , dropout_rate = dropout_rate)
                parameter_loss[(num_layers, num_units, dropout_rate)] = end_vals
                print(num_layers, num_units, dropout_rate)
                print(end_vals['val_loss'])

    return parameter_loss

print(test_hyperparameters())

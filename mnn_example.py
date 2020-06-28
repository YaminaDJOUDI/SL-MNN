import functools
import csv
import tensorflow as tf
import numpy as np
from tensorflow.python.keras import Input
from custom_models.proposed_model import ModelGenerator, HL_MNNGenerator, MNNGenerator
from tensorflow.python.keras.layers import Dense

# Importing dataset
cifar10 = tf.keras.datasets.cifar10
(x_train, y_train), (x_test, y_test) = cifar10.load_data()

x_train = x_train / 255.0
x_test = x_test / 255.0

# flat dimension is needed for Input class
flat_dim = 32 * 32 * 3

x_train = np.reshape(x_train, (50000, flat_dim))
x_test = np.reshape(x_test, (10000, flat_dim))
split = 0.1
x_train, x_val = x_train[:int(len(x_train) * (1 - split))], x_train[-int(len(x_train) * split):]
y_train, y_val = y_train[:int(len(y_train) * (1 - split))], y_train[-int(len(y_train) * split):]


input_layer = Input(shape=(flat_dim,), name='input')
mnn = MNNGenerator(input_layer, 10)

# OvR sub neural network of the Airplane class
snn_layers = [Dense(51, activation='sigmoid'), Dense(51, activation='sigmoid'),
              Dense(1, activation='sigmoid', name='OvR_out')]

sub_model_labels = {0: 1, 1: 0, 2: 0, 3: 0, 4: 0, 5: 0, 6: 0, 7: 0, 8: 0, 9: 0}
mnn.add_sub_model(layers=snn_layers, name="model_OvR",
                  loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
                  labels=sub_model_labels)

snn_layers = [Dense(51, activation='sigmoid'), Dense(51, activation='sigmoid'),
              Dense(1, activation='sigmoid', name='OvO_out')]

# OvO sub neural network of the Airplane and Automobile class
sub_model_labels = {0: 0, 1: 1, 2: 2, 3: 2, 4: 2, 5: 2, 6: 2, 7: 2, 8: 2, 9: 2}
mnn.add_sub_model(layers=snn_layers, name="model_OvO",
                  loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
                  labels=sub_model_labels)

# Superclass sub neural network of Animals and Vehicles
sub_model_labels = {0: 1, 1: 1, 2: 0, 3: 0, 4: 0, 5: 0, 6: 0, 7: 0, 8: 1, 9: 1}
snn_layers = [Dense(51, activation='sigmoid'), Dense(51, activation='sigmoid'),
              Dense(1, activation='sigmoid', name='Sc_out')]
mnn.add_sub_model(layers=snn_layers, name="model_SuperClass",
                  loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
                  labels=sub_model_labels)

optimizer = tf.keras.optimizers.SGD(momentum=0.9, nesterov=True)
final_layers = [Dense(512, activation='sigmoid'), Dense(10, activation='softmax', name='final_output')]
mnn.finalize_model(final_layers, name="experiment1_model")

mnn.compile_sub_models(optimizer=optimizer, metrics=["accuracy"])
mnn.fit_sub_models(x_train, y_train, epochs=150,
                   batch_size=320, validation_data=(x_val, y_val))
mnn.final_model.compile(optimizer=optimizer,
                        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                        metrics=["accuracy"])
mnn.final_model.fit(x_train, y_train, epochs=100, batch_size=320, validation_data=(x_val, y_val))

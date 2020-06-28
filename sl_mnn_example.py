import functools
import csv
import tensorflow as tf
import numpy as np
from tensorflow.python.keras import Input
from custom_models.proposed_model import ModelGenerator
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

def loss(model, x, y, training, loss_object):
    # training=training is needed only if there are layers with different
    # behavior during training versus inference (e.g. Dropout).
    y_ = model(x, training=training)
    return loss_object(y_true=y, y_pred=y_)


# Gets the loss value and gradient during training
def grad(model, inputs, targets, training, loss_object):
    with tf.GradientTape() as tape:
        loss_value = loss(model, inputs, targets, training, loss_object)
    return loss_value, tape.gradient(loss_value, model.trainable_variables)


# Weighted loss function for balance
# Positive weight is the weight that label 1 has
# Negative weight is the weight that label 0 has
# If a label is -1 the weight is 0
def custom_binary_loss(positive_weight, negative_weight, y_true, y_pred):
    def get_weight(num):
        if num == -1:
            return 0
        elif num == 0:
            return negative_weight
        elif num == 1:
            return positive_weight
    weights = [get_weight(i) for i in list(y_true)]
    loss_object = tf.keras.losses.BinaryCrossentropy(from_logits=True)
    return loss_object(y_true=y_true, y_pred=y_pred, sample_weight=weights)


def combine_weighted(weight1, weight2, gradient1, gradient2):
    # Ratio is the amount gradient1 should count 1-ratio is the amount gradient2 should count
    # Gradient 1 is of the intermediary network, Gradient 2 is of the sub network
    gradient1 = tf.multiply(gradient1, weight1)
    gradient2 = tf.multiply(gradient2, weight2)
    return tf.add(gradient1, gradient2)


# Gets the gradients of all the models
def get_all_gradients_v1(slmnn, x, y_index, label_array_dict, loss_dict, training=True):
    gradients = []
    name = slmnn.models[-1].name
    y = label_array_dict[name][y_index]
    full_loss_value, full_grad = grad(slmnn.models[-1], x, y, training, loss_dict[name])
    for model in slmnn.models[:-1]:
        y = label_array_dict[model.name][y_index]
        model_loss_value, model_grad = grad(model, x, y, training,
                                            loss_dict[model.name])
        gradients.append((model_loss_value, model_grad))
    gradients.append((full_loss_value, full_grad))
    return gradients


# (Graph_index, layer_index) -> (Graph_index, layer_index)
# Combines and applies all gradients
def apply_all_gradients_v2(slmnn, combine_func, gradients, gradient_dict, optimizer):
    # For every sub model
    for g_i in range(len(gradients) - 1):

        # For every layers gradients
        # l_i / 2 is the index of the layer in model.layers
        # l_i is the index of the gradient of the weights of this layer
        # l_i + 1 is the index of the gradient of the biases of this layer

        for l_i in range(0, len(gradients[g_i][1]), 2):
            g = gradients[g_i][1]
            layer_gradient_weights = g[l_i]
            layer_gradient_biases = g[l_i + 1]

            # If multiple models have a gradient for this layer
            if (g_i, l_i) in gradient_dict.keys():
                full_l = gradient_dict.get((g_i, l_i))
                gradients[-1][1][full_l] = combine_func(gradients[-1][1][full_l], layer_gradient_weights)
                gradients[-1][1][full_l + 1] = combine_func(gradients[-1][1][full_l + 1], layer_gradient_biases)

            # If only one model has a gradient for this layer.
            else:
                layer = slmnn.models[g_i].layers[int(l_i / 2) + 1]
                optimizer.apply_gradients(
                    zip([layer_gradient_weights, layer_gradient_biases], layer.trainable_weights))
    optimizer.apply_gradients(zip(gradients[-1][1], slmnn.models[-1].trainable_weights))


sub_loss = functools.partial(custom_binary_loss, 9, 1)
full_loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

input_layer = Input(shape=(flat_dim,), name='input')
slmnn = ModelGenerator(input_layer, 10)

# OvR sub neural network of the Airplane class
snn_layers = [Dense(51, activation='sigmoid'), Dense(51, activation='sigmoid'),
              Dense(1, activation='sigmoid', name='OvR_out')]

sub_model_labels = {0: 1, 1: 0, 2: 0, 3: 0, 4: 0, 5: 0, 6: 0, 7: 0, 8: 0, 9: 0}
slmnn.add_sub_model(layers=snn_layers, name="model_OvR",
                    loss=sub_loss,
                    labels=sub_model_labels)

snn_layers = [Dense(51, activation='sigmoid'), Dense(51, activation='sigmoid'),
              Dense(1, activation='sigmoid', name='OvR_out')]

# OvO sub neural network of the Airplane and Automobile class
sub_model_labels = {0: 0, 1: 1, 2: -1, 3: -1, 4: -1, 5: -1, 6: -1, 7: -1, 8: -1, 9: -1}
slmnn.add_sub_model(layers=snn_layers, name="model_OvO",
                    loss=functools.partial(custom_binary_loss, 1, 1),
                    labels=sub_model_labels)

# Superclass sub neural network of Animals and Vehicles
sub_model_labels = {0: 1, 1: 1, 2: 0, 3: 0, 4: 0, 5: 0, 6: 0, 7: 0, 8: 1, 9: 1}
slmnn.add_sub_model(layers=snn_layers, name="model_SuperClass",
                    loss=functools.partial(custom_binary_loss, 6, 4),
                    labels=sub_model_labels)


final_layers = [Dense(512, activation='sigmoid'), Dense(10, activation='softmax', name='final_output')]

slmnn.finalize_model(final_layers,
                     loss=full_loss,
                     name="experiment1_model",
                     train_labels=y_train)

combine_func = functools.partial(combine_weighted, 0.5, 0.5)
batch_size = 320
optimizer = tf.keras.optimizers.SGD(momentum=0.9, nesterov=True)

early_stopping = tf.keras.callbacks.EarlyStopping(patience=25, min_delta=0, restore_best_weights=True)
early_stopping.set_model(slmnn.models[-1])
early_stopping.on_train_begin()

epoch = 0
while not slmnn.models[-1].stop_training:
    current_index = 0
    amount_of_models = len(slmnn.models)
    print()
    print("Starting epoch " + str(epoch + 1))
    epoch_loss_averages = [tf.keras.metrics.Mean() for i in range(amount_of_models)]
    epoch_accuracy = tf.keras.metrics.SparseCategoricalAccuracy()

    # Loops over training images in batches
    for left_index in range(0, len(x_train), batch_size):
        right_index = left_index + batch_size
        if right_index > len(x_train):
            right_index = len(x_train) - 1

        gradients = get_all_gradients_v1(slmnn=slmnn, x=x_train[left_index:right_index],
                                         y_index=list(range(left_index, right_index)),
                                         label_array_dict=slmnn.label_array_dict, loss_dict=slmnn.loss_dict)

        apply_all_gradients_v2(slmnn=slmnn, combine_func=combine_func, gradients=gradients,
                               gradient_dict=slmnn.gradient_dict, optimizer=optimizer)

        for i in range(amount_of_models):
            # Update metrics
            model_labels = slmnn.label_array_dict[slmnn.models[i].name][
                           left_index:right_index]
            model_prediction = slmnn.models[i](x_train[left_index:right_index])
            epoch_loss_averages[i].update_state(gradients[i][0])
            if i == amount_of_models - 1:
                epoch_accuracy.update_state(model_labels, model_prediction)

        if right_index % 6400 == 0:
            # Show some metrics during epoch
            print()
            for i in range(amount_of_models):
                print(slmnn.models[i].name +
                      " Batch {:03d}: Loss: {:.3f}".format(int(right_index / batch_size),
                                                                             epoch_loss_averages[i].result()))
            print("Accuracy: {:.3%}".format(epoch_accuracy.result()))
    # Get val_loss
    val_loss = loss(slmnn.models[-1], x_val, y_val,
                    loss_object=slmnn.loss_dict[slmnn.models[-1].name], training=False)

    # Get val_acc
    val_acc = tf.keras.metrics.Accuracy()
    logits = slmnn.models[-1](x_val, training=False)
    prediction = tf.argmax(logits, axis=1, output_type=tf.int32)
    val_acc(prediction, y_val)

    # Callbacks
    early_stopping.on_epoch_end(epoch=epoch, logs={'val_loss': val_loss})
    val_loss = float(val_loss)
    print("Validation loss : " + str(val_loss))
    val_acc = float(val_acc.result())
    print("Validation accuracy : " + str(val_acc))
    epoch += 1

early_stopping.on_train_end()
print("Training done!")

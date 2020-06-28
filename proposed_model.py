import tensorflow as tf
import numpy as np
from tensorflow.python.keras.models import Model


class ModelGenerator():
    def __init__(self, input_layer, number_of_classes):
        self.input_layer = input_layer
        self.number_of_classes = number_of_classes
        self.models = []
        self.merging_layer = []
        self.gradient_dict = {}
        self.loss_dict = {}
        self.label_dict = {}
        self.label_array_dict = {}

    def finalize_dicts(self, train_labels):
        self.label_array_dict = self.create_label_array_dict(train_labels)
        self.gradient_dict = self.create_gradient_dict()

    def create_label_array_dict(self, train_labels):
        # Creates a dict which determines which models should have which labels
        temp_dict = {}
        for key in self.label_dict.keys():
            labels = self.label_dict[key]
            if len(labels.keys()) > 0:
                temp_arr = [labels[x[0]] for x in list(train_labels)]
                temp_arr = np.asarray(temp_arr).astype('float64').reshape((-1, 1))
                temp_dict[key] = temp_arr
            else:
                temp_dict[key] = train_labels
        return temp_dict

    def create_gradient_dict(self):
        # Creates a dict which determines which gradients to add to each other
        temp_dict = {}
        full_model = self.models[-1]
        gradient_index = 0
        for layer in full_model.layers:
            if len(layer.weights) > 0:
                for m in range(len(self.models) - 1):
                    model = self.models[m]
                    l_i = 0
                    for l in range(len(model.layers)):
                        if model.layers[l] is layer:
                            # If the gradients point to the same layer
                            temp_dict[(m, l_i)] = gradient_index
                            break
                        if len(model.layers[l].weights) > 0:
                            l_i += 2
                    # Go back to the very first for loop
                    else:
                        continue
                    break
                # Every 2 indeces is the a new layer's weights, this number + 1 is the biases of these layers.
                gradient_index += 2

        # (model, layer) -> full_layer
        return temp_dict

    def add_sub_model(self, layers, name, loss, labels, binary=True):
        x = self.input_layer
        for l in range(len(layers) - 1):
            layers[l]._name = name + str(l)
            x = layers[l](x)
        self.merging_layer.append(x)
        self.label_dict[name] = labels
        self.loss_dict[name] = loss
        sub_model_output = layers[-1](x)
        self.models.append(Model(inputs=self.input_layer, outputs=sub_model_output, name=name))

    def finalize_model(self, layers, loss, name, train_labels):
        # Merge all the layers of the sub models to one and connect to the final layers
        x = tf.keras.layers.concatenate(self.merging_layer)
        counter = 0
        for layer in layers:
            layer._name = name + str(counter)
            x = layer(x)
            counter += 1
        self.label_dict[name] = {}
        self.loss_dict[name] = loss
        self.models.append(Model(inputs=self.input_layer, outputs=x, name=name))
        self.finalize_dicts(train_labels)


class MNNGenerator():
    def __init__(self, input_layer, number_of_classes):
        self.input_layer = input_layer
        self.number_of_classes = number_of_classes
        self.models = []
        self.merging_layer = []
        self.loss_dict = {}
        self.label_dict = {}
        self.label_array_dict = {}
        self.final_model = None

    def add_sub_model(self, layers, name, loss, labels, binary=True):
        x = self.input_layer
        for l in range(len(layers)):
            x = layers[l](x)
        self.merging_layer.append(x)
        self.label_dict[name] = labels
        self.loss_dict[name] = loss
        self.models.append(Model(inputs=self.input_layer, outputs=x, name=name))

    def update_loss(self, loss_dict):
        for key in loss_dict.keys():
            if key in self.loss_dict.keys():
                self.loss_dict[key] = loss_dict[key]

    def create_label_array_dict(self, train_labels):
        # Creates a dict which determines which models should have which labels
        temp_dict = {}
        for key in self.label_dict.keys():
            labels = self.label_dict[key]
            if len(labels.keys()) > 0:
                temp_arr = [labels[x[0]] for x in list(train_labels)]
                temp_arr = np.asarray(temp_arr).astype('float64').reshape((-1, 1))
                temp_dict[key] = temp_arr
            else:
                temp_dict[key] = train_labels
        return temp_dict

    def compile_sub_models(self, **kwargs):
        for sub_model in self.models:
            sub_model.compile(**kwargs, loss=self.loss_dict[sub_model.name])

    def set_models_trainable(self, trainable):
        for sub_model in self.models:
            sub_model.trainable = trainable

    def fit_sub_models(self, x_train, y_train, **kwargs):
        y_train_dict = self.create_label_array_dict(y_train)
        y_val_dict = None
        if 'validation_data' in kwargs.keys():
            (x_val, y_val) = kwargs['validation_data']
            y_val_dict = self.create_label_array_dict(y_val)
        for sub_model in self.models:
            neg_count = 0
            pos_count = 0
            for key in self.label_dict[sub_model.name].keys():
                item = self.label_dict[sub_model.name][key]
                if item == 1:
                    pos_count += 1
                elif item == 0:
                    neg_count += 1
            pos_weight = neg_count
            neg_weight = pos_count
            class_weights = {0: neg_weight, 1: pos_weight, 2: 0}
            if y_val_dict is not None:
                (x_val, y_val) = kwargs['validation_data'][0], kwargs['validation_data'][1]
                val_weights = np.asarray([class_weights[int(c[0])] for c in y_val_dict[sub_model.name]])
                kwargs['validation_data'] = (x_val, y_val_dict[sub_model.name], val_weights)
            escb = tf.keras.callbacks.EarlyStopping(patience=10, min_delta=0.0001, restore_best_weights=True)
            sub_model.fit(x_train, y_train_dict[sub_model.name], callbacks=[escb], class_weight=class_weights, **kwargs)
            sub_model.trainable = False

    def finalize_model(self, layers, name):
        # Merge all the layers of the sub models to one and connect to the final layers
        x = tf.keras.layers.concatenate(self.merging_layer)
        count = 0
        for layer in layers:
            layer._name = name + str(count)
            x = layer(x)
            count += 1
        self.final_model = Model(inputs=self.input_layer, outputs=x, name=name)
        return self.final_model

    def compile_final_model(self, **kwargs):
        self.final_model.compile(**kwargs)

    def fit_final_model(self, **kwargs):
        self.final_model.fit(**kwargs)


class HL_MNNGenerator():
    def __init__(self, input_layer, number_of_classes):
        self.input_layer = input_layer
        self.number_of_classes = number_of_classes
        self.models = []
        self.merging_layer = []
        self.loss_dict = {}
        self.label_dict = {}
        self.label_array_dict = {}
        self.final_model = None

    def add_sub_model(self, layers, name, loss, labels, binary=True):
        x = self.input_layer
        for l in range(len(layers) - 1):
            x = layers[l](x)
        self.merging_layer.append(x)
        x = layers[-1](x)
        self.label_dict[name] = labels
        self.loss_dict[name] = loss
        self.models.append(Model(inputs=self.input_layer, outputs=x, name=name))

    def update_loss(self, loss_dict):
        for key in loss_dict.keys():
            if key in self.loss_dict.keys():
                self.loss_dict[key] = loss_dict[key]

    def create_label_array_dict(self, train_labels):
        # Creates a dict which determines which models should have which labels
        temp_dict = {}
        for key in self.label_dict.keys():
            labels = self.label_dict[key]
            if len(labels.keys()) > 0:
                temp_arr = [labels[x[0]] for x in list(train_labels)]
                temp_arr = np.asarray(temp_arr).astype('float64').reshape((-1, 1))
                temp_dict[key] = temp_arr
            else:
                temp_dict[key] = train_labels
        return temp_dict

    def compile_sub_models(self, **kwargs):
        for sub_model in self.models:
            sub_model.compile(**kwargs, loss=self.loss_dict[sub_model.name])

    def set_models_trainable(self, trainable):
        for sub_model in self.models:
            sub_model.trainable = trainable

    def fit_sub_models(self, x_train, y_train, **kwargs):
        y_train_dict = self.create_label_array_dict(y_train)
        y_val_dict = None
        if 'validation_data' in kwargs.keys():
            (x_val, y_val) = kwargs['validation_data']
            y_val_dict = self.create_label_array_dict(y_val)
        for sub_model in self.models:
            neg_count = 0
            pos_count = 0
            for key in self.label_dict[sub_model.name].keys():
                item = self.label_dict[sub_model.name][key]
                if item == 1:
                    pos_count += 1
                elif item == 0:
                    neg_count += 1
            pos_weight = neg_count
            neg_weight = pos_count
            class_weights = {0: neg_weight, 1: pos_weight, 2: 0}
            if y_val_dict is not None:
                (x_val, y_val) = kwargs['validation_data'][0], kwargs['validation_data'][1]
                val_weights = np.asarray([class_weights[int(c[0])] for c in y_val_dict[sub_model.name]])
                kwargs['validation_data'] = (x_val, y_val_dict[sub_model.name], val_weights)
            escb = tf.keras.callbacks.EarlyStopping(patience=10, min_delta=0.0001, restore_best_weights=True)
            sub_model.fit(x_train, y_train_dict[sub_model.name], callbacks=[escb], class_weight=class_weights, **kwargs)
            sub_model.trainable = False

    def finalize_model(self, layers, name):
        # Merge all the layers of the sub models to one and connect to the final layers
        x = tf.keras.layers.concatenate(self.merging_layer)
        count = 0
        for layer in layers:
            layer._name = name + str(count)
            x = layer(x)
            count += 1
        self.final_model = Model(inputs=self.input_layer, outputs=x, name=name)
        return self.final_model

    def compile_final_model(self, **kwargs):
        self.final_model.compile(**kwargs)

    def fit_final_model(self, **kwargs):
        self.final_model.fit(**kwargs)

from typing import List, Dict
import numpy as np


class Layer(object):

    def __init__(self, weights: np.array, bias: np.array):
        self.weights = weights
        self.bias = bias
        self.activation_cache = None
        self.previous_linear_activation_cache = None

    def linear_forward(self, activation: np.array, weights: np.array, bias: np.array) -> np.array:
        '''

        :param activation: the activations of the previous layer
        :param weights: the weight matrix of the current layer (shape [size of current layer, size of previous layer])
        :param bias: bias vector of the current layer (shape [size of current layer, 1])
        :return: the linear component of the activation function (the value before applying the non-linear function)
        '''
        self.previous_linear_activation_cache = activation
        self.weights = weights
        self.bias = bias

        matrix = np.concatenate((weights, bias), axis=1)
        activation_extended = np.insert(activation, len(activation), 1.0, axis=0)
        return matrix.dot(activation_extended)


    # def soft_max(self, z: np.array):
    #     '''
    #
    #     :param z: the linear component of the activation function
    #     :return: activations of the layer
    #     '''
    #     self.activation_cache = z
    #     z_exp = np.exp(z)
    #     sum_z_exp = np.sum(z_exp)
    #     return z_exp / sum_z_exp

    def soft_max(self, z: np.array):
        """

        :param z: the linear component of the activation function. it can be matrix each column represents sample
        :return: activations of the layer
        """
        z_exp = np.exp(z)
        return z_exp / np.sum(z_exp, axis=0)

    def relu(self, z: np.array) -> np.array:
        '''

        :param z: the linear component of the activation function
        :return: activations of the layer
        '''
        self.activation_cache = z
        # result = np.maximum(np.zeros(len(z)), z)
        result = np.maximum(np.zeros(z.shape), z)
        return result

    def linear_activation_forward(self, activations_prev: np.array, weights: np.array, bias: np.array,
                                  activation: str) -> np.array:
        '''

        :param activations_prev: activations of the previous layer
        :param weights: the weight matrix of the current layer
        :param bias: bias vector of the current layer
        :param activation: activation function to be used (“softmax” or “relu”)
        :return: activations of the current layer
        '''
        z = self.linear_forward(activations_prev, weights, bias)
        if activation == "relu":
            return self.relu(z)
        if activation == "softmax":
            return self.soft_max(z)

    def compute_cost(self, label_predictions: np.ndarray, y: np.ndarray, epsilon=1e-12) -> np.ndarray:
        '''

        :param epsilon: to avoid log(0)
        :param label_predictions: probability vector corresponding label predictions, shape (num_of_classes, number of examples)
        :param y: the labels vector, shape (num_of_classes, number of examples)
        :return: the cross-entropy cost for all inputs
        '''
        label_predictions = np.clip(label_predictions, epsilon, 1)
        n_samples = label_predictions.shape[1]
        return -np.sum(y * np.log(label_predictions)) / n_samples


class Network(object):

    def __init__(self):
        self.index_2_layer = {}
        self.last_layer = None

    def initialize_parameters(self, layer_dims: List[int]) -> Dict[int, Layer]:
        '''

        :param layer_dims: an array of the dimensions of each layer in the network
        :return: a dictionary containing the initialized W and b parameters of each layer (W1…WL, b1…bL).
        '''

        index_2_layer = {}
        for layer in range(1, len(layer_dims), 1):
            weight_layer_shape = (layer_dims[layer], layer_dims[layer - 1])
            weight_matrix = np.zeros(weight_layer_shape) + np.random.uniform(-1.0, 1.0, weight_layer_shape)
            bias_shape = (layer_dims[layer], 1)
            bias_vector = np.zeros(bias_shape) + np.random.uniform(-1.0, 1.0, bias_shape)
            index_2_layer[layer] = Layer(weight_matrix, bias_vector)

        self.index_2_layer = dict(index_2_layer)
        self.last_layer = self.index_2_layer.popitem()[1]
        return index_2_layer

    def linear_model_forward(self, x: np.array, use_batchnorm: bool = False) -> np.array:
        """

        :param x: the data, numpy array of shape (input size, number of examples)
        :param parameters: the initialized W and b parameters of each layer
        :param use_batchnorm: boolean flag used to determine whether to apply batchnorm after the activation
        :return: last post-activation value
        """
        prev_activations = x
        for index, layer in self.index_2_layer.items():
            prev_activations = layer.linear_activation_forward(prev_activations, layer.weights, layer.bias, "relu")

        result = self.last_layer.linear_activation_forward(prev_activations, self.last_layer.weights, self.last_layer.bias, "softmax")
        return result



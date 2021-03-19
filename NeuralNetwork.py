from typing import List, Dict, Tuple
import numpy as np


class Layer(object):

    def __init__(self, weights: np.array, bias: np.array, activation="relu"):
        self.weights = weights
        self.bias = bias
        self.activation_cache = None
        self.previous_linear_activation_cache = None
        self.activation_func = activation

    def linear_forward(self, activation: np.array, weights: np.array, bias: np.array) -> np.array:
        '''
        Calculates the linear part of forward propagation
        :param activation: the activations of the previous layer
        :param weights: the weight matrix of the current layer (shape [size of current layer, size of previous layer])
        :param bias: bias vector of the current layer (shape [size of current layer, 1])
        :return: the linear component of the activation function (the value before applying the non-linear function)
        '''
        self.previous_linear_activation_cache = activation
        self.weights = weights
        self.bias = bias

        matrix = np.concatenate((weights, bias), axis=1) # [W|b]
        activation_extended = np.insert(activation, len(activation), 1.0, axis=0) # [A|1]
        return matrix.dot(activation_extended) # [W|b]*[A|1]

    def soft_max(self, z: np.array):
        """
        Softmax activation function
        :param z: the linear component of the activation function. it can be matrix each column represents sample
        :return: activations of the layer
        """
        z_exp = np.exp(z)
        return z_exp / np.sum(z_exp, axis=0)

    def relu(self, z: np.array) -> np.array:
        '''
        ReLU activation function
        :param z: the linear component of the activation function
        :return: activations of the layer
        '''
        self.activation_cache = z
        result = np.maximum(np.zeros(z.shape), z)
        return result

    def linear_activation_forward(self, activations_prev: np.array, weights: np.array, bias: np.array,
                                  activation: str) -> np.array:
        '''
        Forward propagation for the LINEAR->ACTIVATION layer
        :param activations_prev: activations of the previous layer
        :param weights: the weight matrix of the current layer
        :param bias: bias vector of the current layer
        :param activation: activation function to be used (“softmax” or “relu”)
        :return: activations of the current layer
        '''
        # TODO: return joined cache
        z = self.linear_forward(activations_prev, weights, bias)
        if self.activation_func == "relu":
            return self.relu(z)
        if self.activation_func == "softmax":
            return self.soft_max(z)

    def compute_cost(self, label_predictions: np.ndarray, y: np.ndarray, epsilon=1e-12) -> np.ndarray:
        '''
        Calculate cost function - categorical cross-entropy loss
        :param epsilon: to avoid log(0)
        :param label_predictions: probability vector corresponding label predictions, shape (num_of_classes, number of examples)
        :param y: the labels vector, shape (num_of_classes, number of examples)
        :return: the cross-entropy cost for all inputs
        '''
        label_predictions = np.clip(label_predictions, epsilon, 1)
        n_samples = label_predictions.shape[1]
        return -np.sum(y * np.log(label_predictions)) / n_samples

    def apply_batchnorm(self, activations: np.array, epsilon=1e-12) -> np.array:
        '''
        Prior activation batch normalization
        :param activations: the activation values of the layer
        :return: normalized activation values
        '''
        m = activations.shape[1]
        mu = np.sum(activations, axis=1)/m
        var = np.sum((activations - np.vstack(mu))**2)/m
        normalized_activations = (activations - np.vstack(mu))/np.sqrt(var + epsilon)
        return normalized_activations

    def linear_backward(self, dz: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """

        :param dz: the gradient of the cost with respect to the linear output of the current layer dz:=[dL/dz]
        :param cache: tuple of values (A_prev, W, b) coming from the forward propagation in the current layer
        :return:
                dA_prev -- Gradient of the cost with respect to the activation (of the previous layer l-1), same shape as A_prev
                dW -- Gradient of the cost with respect to W (current layer l), same shape as W
                db -- Gradient of the cost with respect to b (current layer l), same shape as b

        """
        n_samples = self.previous_linear_activation_cache.shape[1]

        # Z=W*A + b
        # dW:=dL/dW = 1/n_samples * (dz*A)
        # db:=dL/db = 1/n_samples * sum(dz[i])
        # dA_prev:=dL/dA_prev = W*dz
        dW = 1.0 / n_samples * np.dot(dz, self.previous_linear_activation_cache.T)
        db = 1.0 / n_samples * np.sum(dz, axis=1, keepdims=True)
        dA_prev = np.dot(self.weights.T, dz)
        return dA_prev, dW, db

    def linear_activation_backward(self, da: np.ndarray):
        """

        :param da: post activation gradient of the current layer
        :return:
                dA_prev -- Gradient of the cost with respect to the activation (of the previous layer l-1), same shape as A_prev
                dW -- Gradient of the cost with respect to W (current layer l), same shape as W
                db -- Gradient of the cost with respect to b (current layer l), same shape as b
        """

        # cache is stored in Layer's fields see CTOR
        da_prev, dw, db = None, None, None
        if self.activation_func == "relu":
            dz = self.__relu_backward(da)
            da_prev, dw, db = self.linear_backward(dz)
        # elif self.activation_func == "softmax":
        #     dz = self.__softmax_backward(da)
        #     da_prev, dw, db = self.linear_backward(dz)

        return da_prev, dw, db

    def __softmax_backward(self, da: np.ndarray) -> np.ndarray:
        pass

    def __relu_backward(self, da: np.ndarray) -> np.ndarray:
        """

        :param da: the post-activation gradient
        :return dz: Gradient of the cost with respect to Z
        """

        # relu(z)=a
        # dL/dz = dL/da * da/dz = da * relu'(z)

        z = self.activation_cache
        dz = np.array(da, copy=True)
        dz[z <= 0] = 0
        assert (dz.shape == z.shape)
        return dz


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
            weight_matrix = np.random.uniform(-1.0, 1.0, weight_layer_shape) # this should be randoms only, for some reason he advices to use randn for nornal distribution
            bias_shape = (layer_dims[layer], 1)
            bias_vector = np.zeros(bias_shape) # this should be zeros only
            index_2_layer[layer] = Layer(weight_matrix, bias_vector)

        self.index_2_layer = dict(index_2_layer)
        self.last_layer = self.index_2_layer.popitem()[1]
        self.last_layer.activation_func = "softmax"
        return index_2_layer

    def linear_model_forward(self, x: np.array, use_batchnorm: bool = False) -> np.array:
        """

        :param x: the data, numpy array of shape (input size, number of examples)
        :param parameters: the initialized W and b parameters of each layer
        :param use_batchnorm: boolean flag used to determine whether to apply batchnorm after the activation
        :return: last post-activation value
        """
        # TODO: add "parameters" arg

        prev_activations = x
        for index, layer in self.index_2_layer.items():
            prev_activations = layer.linear_activation_forward(prev_activations, layer.weights, layer.bias, "relu")
            if use_batchnorm:
                prev_activations = layer.apply_batchnorm(prev_activations)

        result = self.last_layer.linear_activation_forward(prev_activations, self.last_layer.weights, self.last_layer.bias, "softmax")
        return result



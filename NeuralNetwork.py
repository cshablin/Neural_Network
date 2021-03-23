from typing import List, Dict, Tuple
import numpy as np

SOFTMAX = "softmax"
RELU = "relu"


class Layer(object):
    def __init__(self, weights: np.array, bias: np.array, activation):
        self.weights = weights
        self.bias = bias
        self.activation_func = activation


class LinearCache(object):
    def __init__(self, weights: np.array, bias: np.array, activation: np.array):
        self.weights = weights
        self.bias = bias
        self.activation = activation


def initialize_parameters(layer_dims: List[int]) -> Dict[int, Layer]:
    """
    Initialize weights and biases for all of the layers of the network
    :param layer_dims: an array of the dimensions of each layer in the network
    :return: a dictionary containing the initialized W and b parameters of each layer (W1…WL, b1…bL).
    """
    index_2_layer = {}

    for layer in range(1, len(layer_dims), 1):
        weight_layer_shape = (layer_dims[layer], layer_dims[layer - 1])
        weight_matrix = np.random.uniform(-1.0, 1.0,
                                          weight_layer_shape)  # this should be randoms only, for some reason he advices to use randn for nornal distribution
        bias_shape = (layer_dims[layer], 1)
        bias_vector = np.zeros(bias_shape)  # this should be zeros only
        activation_function = SOFTMAX if layer == len(layer_dims) else RELU
        index_2_layer[layer] = Layer(weight_matrix, bias_vector, activation_function)

    return index_2_layer


def linear_forward(activation: np.array, weights: np.array, bias: np.array) -> Tuple[np.array, LinearCache]:
    """
    Calculates the linear part of forward propagation
    :param activation: the activations of the previous layer
    :param weights: the weight matrix of the current layer (shape [size of current layer, size of previous layer])
    :param bias: bias vector of the current layer (shape [size of current layer, 1])
    :return: tuple of the linear component of the activation function (the value before applying the non-linear function) and the linear cache
    """

    matrix = np.concatenate((weights, bias),
                            axis=1)  # [W|b], W.shape = [curr_layer, prev_layer] (already transposed)
    activation_extended = np.insert(activation, len(activation), 1.0, axis=0)  # [A|1]
    return matrix.dot(activation_extended), LinearCache(weights, bias, activation)  # z = [W|b]*[A|1], LinearCache


def soft_max(z: np.array) -> Tuple[np.array, np.array]:
    """
    Softmax activation function
    :param z: the linear component of the activation function. it can be matrix each column represents sample
    :return: tuple of activations of the layer and activation cache
    """

    z_exp = np.exp(z)
    return z_exp / np.sum(z_exp, axis=0), z


def relu(z: np.array) -> Tuple[np.array, np.array]:
    """
    ReLU activation function
    :param z: the linear component of the activation function
    :return: tuple of activations of the layer and activation cache
    """

    a = np.maximum(np.zeros(z.shape), z)
    return a, z


def linear_activation_forward(activations_prev: np.array, weights: np.array, bias: np.array,
                              activation: str) -> Tuple[np.array, Dict[str, Tuple[LinearCache, np.array]]]:
    """
    Forward propagation for the LINEAR->ACTIVATION layer
    :param activations_prev: activations of the previous layer
    :param weights: the weight matrix of the current layer
    :param bias: bias vector of the current layer
    :param activation: activation function to be used (“softmax” or “relu”)
    :return: tuple of activations of the current layer and a dictionary of linear_cache and activation_cache
    """
    z, linear_cache = linear_forward(activations_prev, weights, bias)
    if activation == RELU:
        a, activation_cache = relu(z)
    elif activation == SOFTMAX:
        a, activation_cache = soft_max(z)
    else:
        raise RuntimeError('Unknown activation function')
    cache = {'linear_cache': linear_cache, 'activation_cache': activation_cache}
    return a, cache


def linear_model_forward(x: np.array, parameters: Dict[int, Layer], use_batchnorm: bool = False) -> Tuple[
    np.array, Dict]:
    """
    Forward propagation for all of the network's layers and applying batchnorm if needed
    :param x: the data, numpy array of shape (input size, number of examples)
    :param parameters: the initialized W and b parameters of each layer
    :param use_batchnorm: boolean flag used to determine whether to apply batchnorm after the activation
    :return: last post-activation value
    """

    a = x
    caches = {}
    for index, layer in parameters.items():
        (a, cache) = linear_activation_forward(a, layer.weights, layer.bias,
                                               layer.activation_func)
        if use_batchnorm:  # aply batchnorm after activation
            a = apply_batchnorm(a)
        caches[index] = cache

    return a, caches


def compute_cost(label_predictions: np.ndarray, y: np.ndarray, epsilon=1e-12) -> np.ndarray:
    """
    Calculate cost function - categorical cross-entropy loss
    :param epsilon: to avoid log(0)
    :param label_predictions: probability vector corresponding label predictions, shape (num_of_classes, number of examples)
    :param y: the labels vector, shape (num_of_classes, number of examples)
    :return: the cross-entropy cost for all inputs
    """
    label_predictions = np.clip(label_predictions, epsilon, 1)
    n_samples = label_predictions.shape[1]
    return -np.sum(y * np.log(label_predictions)) / n_samples


def apply_batchnorm(activations: np.array, epsilon=1e-12) -> np.array:
    """
    Prior activation batch normalization
    :param epsilon: to avoid division by 0
    :param activations: the activation values of the layer
    :return: normalized activation values
    """
    batch_size = activations.shape[1]
    mu = np.sum(activations, axis=1) / batch_size
    var = np.sum((activations - np.vstack(mu)) ** 2) / batch_size
    normalized_activations = (activations - np.vstack(mu)) / np.sqrt(var + epsilon)
    return normalized_activations


def linear_backward(dz: np.ndarray, cache: LinearCache) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Calculates a linear part of the backward propagation process for a single layer
    :param dz: the gradient of the cost with respect to the linear output of the current layer dz:=[dL/dz]
    :param cache: tuple of values (A_prev, W, b) coming from the forward propagation in the current layer
    :return:
            dA_prev -- Gradient of the cost with respect to the activation (of the previous layer l-1), same shape as A_prev
            dW -- Gradient of the cost with respect to W (current layer l), same shape as W
            db -- Gradient of the cost with respect to b (current layer l), same shape as b

    """
    n_samples = cache.activation.shape[1]

    # Z=W*A + b
    # dW:=dL/dW = 1/n_samples * (dz*A)
    # db:=dL/db = 1/n_samples * sum(dz[i])
    # dA_prev:=dL/dA_prev = W*dz
    dW = 1.0 / n_samples * np.dot(dz, cache.activation.T)
    db = 1.0 / n_samples * np.sum(dz, axis=1, keepdims=True)
    dA_prev = np.dot(cache.weights.T, dz)
    return dA_prev, dW, db


def linear_activation_backward(da: np.ndarray, cache: Dict[str, np.array], activation: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Calculates the backward propagation for the LINEAR->ACTIVATION layer
    :param activation: string for "relu" or "softmax" activations
    :param cache: includes activation_cache and linear_cache from the forward propagation
    :param da: post activation gradient of the current layer
    :return:
            dA_prev -- Gradient of the cost with respect to the activation (of the previous layer l-1), same shape as A_prev
            dW -- Gradient of the cost with respect to W (current layer l), same shape as W
            db -- Gradient of the cost with respect to b (current layer l), same shape as b
    """
    activation_cache = cache["activation_cache"]
    linear_cache = cache["linear_cache"]
    if activation == RELU:
        dz = __relu_backward(da, activation_cache)
        da_prev, dw, db = linear_backward(dz, linear_cache)
    elif activation == SOFTMAX:
        dz = __softmax_backward(da, activation_cache)
        da_prev, dw, db = linear_backward(dz, linear_cache)
    else:
        raise RuntimeError('Unknown activation function')

    return da_prev, dw, db


def __relu_backward(da: np.ndarray, activation_cache: np.array) -> np.ndarray:
    """
    Backward propagation for a ReLU unit
    :param da: the post-activation gradient
    :param activation_cache: holds a forward-propagated activation value
    :return dz: Gradient of the cost with respect to Z
    """

    # relu(z)=a
    # dL/dz = dL/da * da/dz = da * relu'(z)

    z = activation_cache
    dz = np.array(da, copy=True)
    dz[z <= 0] = 0
    assert (dz.shape == z.shape)
    return dz


def __softmax_backward(da: np.ndarray, activation_cache: np.array) -> np.ndarray:
    """
    Backward propagation for a softmax unit
    :param da: delta between prediction and actual value i.e. (predicted - true_value)
    :param activation_cache: holds a forward-propagated activation value
    :return dz: Gradient of the cost with respect to Z
    """

    # softmax(z)=a
    # dL/dz[i] = dL/da * da/dz = da * softmax'(z) = da * (da[i] / dz[j]) = da * -(a[i]a[j]) =...= a[i] - y[i]
    dz = da  # shortcut saves time
    z = activation_cache
    assert (dz.shape == z.shape)
    return dz


def linear_model_backward(al: np.array, y: np.array, caches: Dict[int, Dict[str, np.array]]) -> Dict[
    str, np.array]:
    """
    Implementation of the backward propagation process for the entire network
    :param caches: for each layer holds activation_cache and linear_cache
    :param al: the probabilities vectors, the output of the forward propagation
    :param y: the true labels vector (the "ground truth" - true classifications)
    :return: dictionary of dA, DW, db for each layer
    """

    result = {}
    # Initializing the backpropagation for softmax output layer!!!
    da = al - y
    for index in range(len(caches), 0, -1):
        activation_function = SOFTMAX if index == len(caches) else RELU # ugly, if network parameters are not passed to the function, otherwise layer.activation_function could be used
        da, dw, db = linear_activation_backward(da, caches[index], activation_function)
        result[f'dA{index}'] = da
        result[f'dW{index}'] = dw
        result[f'db{index}'] = db
    return result


def update_parameters(parameters: Dict[int, Layer], grads: Dict[str, np.array], learning_rate=0.001) -> Dict[
    int, Layer]:
    """
    Updates parameters using gradient descent
    :param parameters: a dictionary containing the DNN architecture’s parameters (weights and biases for all of the layers)
    :param grads: a dictionary containing the gradients generated by linear_model_backward
    :param learning_rate: the learning rate used to update the parameters
    :return:
    """
    result = parameters.copy()
    for layer in result.keys():
        result[layer].weights -= learning_rate * grads[f'dW{layer}']
        result[layer].bias -= learning_rate * grads[f'db{layer}']

    return result


def L_layer_model(X, Y, layers_dims, learning_rate=0.01, num_iterations=10, batch_size=100) -> Tuple[
    np.array]:
    """
    Implementation of a L-layer neural network
    :param X: the input data, a numpy array of shape (height*width , number_of_examples)
    :param Y: the “real” labels of the data, a vector of shape (num_of_classes, number of examples)
    :param layers_dims: a list containing the dimensions of each layer, including the input
    :param learning_rate: the learning rate hyper parameter
    :param num_iterations: a number of epochs
    :param batch_size: a number of samples to consider for a single parameters update
    :return:
        parameters – the parameters learnt by the system during the training (the same parameters that were updated in the update_parameters function).
        costs – the values of the cost function (calculated by the compute_cost function). One value is to be saved after each 100 training iterations (e.g. 3000 iterations -> 30 values).
    """

    params = initialize_parameters(layers_dims)
    number_of_samples = X.shape[1]
    publish_L_at_epoch = 99
    costs = []

    for epoch in range(num_iterations):
        for offset in range(0, number_of_samples, batch_size):
            upper_bound = offset + batch_size if (offset + batch_size) <= number_of_samples else offset + (
                    offset + batch_size) % number_of_samples
            x_sub = X[:, offset:upper_bound]
            y_sub = Y[:, offset:upper_bound]
            y_pred, caches = linear_model_forward(x_sub, params)
            L = compute_cost(y_pred, y_sub)
            if epoch > publish_L_at_epoch:
                costs.append(L)
                publish_L_at_epoch += 100
            grads = linear_model_backward(y_pred, y_sub, caches)
            params = update_parameters(params, grads, learning_rate)
    return params, costs

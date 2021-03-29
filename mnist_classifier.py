import unittest
from NeuralNet import *
import time
from tensorflow import keras


NUM_OF_CLASSES = 10
x_train, y_train, x_test, y_test = None, None, None, None


def load_data():
    global x_train, y_train, x_test, y_test
    (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

    return (x_train.reshape(*x_train.shape[:1], -1), y_train), (x_test.reshape(*x_test.shape[:1], -1), y_test)


def split_train_validation(x: np.ndarray, y: np.ndarray, test_ratio=0.2):
    """
    Splits a reshuffled set by given ratio
    :param x: he input data, a numpy array of shape (height*width , number_of_examples)
    :param y: the “real” labels of the data, a vector of shape (num_of_classes, number of examples)
    :param test_ratio: portion of data to be in validation set
    :return:
        x_train: the rest of X after excluding x_validation
        y_train: the rest of y after excluding y_validation
        x_validation: |X|*test_ration randomly chosen samples from X
        y_validation: |y|*test_ration randomly chosen samples from y

    """
    xy_matrix = np.concatenate((x, y.reshape((y.shape[0], 1))),
                               axis=1)  # MemoryError: Unable to allocate array with shape (60000, 794) and data type float64
    np.random.shuffle(xy_matrix)
    xy_val = xy_matrix[:int(xy_matrix.shape[0] * test_ratio)]
    xy_train = xy_matrix[int(xy_matrix.shape[0] * test_ratio):]
    val = np.split(xy_val, [x.shape[1], ], axis=1)
    train = np.split(xy_train, [x.shape[1], ], axis=1)
    return (train[0], train[1]), (val[0], val[1])


def print_scores(params, costs):
    global x_train, y_train, x_test, y_test
    test_score = predict(x_test.T, y_test.T, params)
    val_score = predict(globals.x_val.T, globals.y_val.T, params)
    train_score = predict(x_train.T, y_train.T, params)
    print("final train score-", train_score)
    print("final validation score-", val_score)
    print("final test score-", test_score)
    print(costs)


class MNIST_classifier(unittest.TestCase):
    np.random.seed(101)
    # we use same test train sets for all 3 runs
    global x_train, y_train, x_test, y_test
    (x_train, y_train), (x_test, y_test) = load_data()
    (x_train, y_train), (globals.x_val, globals.y_val) = split_train_validation(x_train,
                                                                                    y_train)  # Do the split but before x_transform, memory consumption
    y_train = one_hot(y_train, NUM_OF_CLASSES)
    y_test = one_hot(y_test, NUM_OF_CLASSES)
    globals.y_val = one_hot(globals.y_val, NUM_OF_CLASSES)
    x_train = transform_x(x_train)
    x_test = transform_x(x_test)
    globals.x_val = transform_x(globals.x_val)
    image_size = x_test.shape[1]
    layers_dims = [image_size, 20, 7, 5, 10]

    def test_neural_net_regular(self):
        globals.use_batch_norm = False
        globals.use_dropout = False
        print("neural_net_regular")
        start = time.time()
        params, costs = L_layer_model(x_train.T, y_train.T, self.layers_dims, num_iterations=100000, batch_size=32,
                                      learning_rate=0.009)
        end = time.time()
        print(f"Execution time [sec]: {end-start}")
        print_scores(params, costs)

    def test_neural_net_batch_norm(self):
        globals.use_batch_norm = True
        globals.use_dropout = False
        print("neural_net_batch_norm")
        start = time.time()
        params, costs = L_layer_model(x_train.T, y_train.T, self.layers_dims, num_iterations=100000, batch_size=64,
                                                learning_rate=0.05)
        end = time.time()
        print(f"Execution time [sec]: {end - start}")
        print_scores(params, costs)

    def test_neural_net_dropout(self):
        globals.use_batch_norm = False
        globals.use_dropout = True
        globals.dropout_keep_probability = 0.9
        print("neural_net_dropout")
        start = time.time()
        params, costs = L_layer_model(x_train.T, y_train.T, self.layers_dims, num_iterations=100000, batch_size=32,
                                      learning_rate=0.009)
        end = time.time()
        print(f"Execution time [sec]: {end - start}")
        print_scores(params, costs)

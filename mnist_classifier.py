import numpy as np
import NeuralNet
from tensorflow import keras

np.random.seed(101)
NUM_OF_CLASSES = 10
x_train, y_train, x_test, y_test = None, None, None, None


def load_data():
    global x_train, y_train, x_test, y_test
    (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

    return (x_train.reshape(*x_train.shape[:1], -1), y_train), (x_test.reshape(*x_test.shape[:1], -1), y_test)


def get_shuffle_validation(x: np.ndarray, y: np.ndarray, test_ratio=0.2):
    """

    :param x: he input data, a numpy array of shape (height*width , number_of_examples)
    :param y: the “real” labels of the data, a vector of shape (num_of_classes, number of examples)
    :param test_ratio: portion of data to be in validation set
    :return:
        x_validation: randomly chosen from X
        y_validation: randomly chosen from Y
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
    test_score = NeuralNet.predict(x_test.T, y_test.T, params)
    val_score = NeuralNet.predict(NeuralNet.x_val.T, NeuralNet.y_val.T, params)
    train_score = NeuralNet.predict(x_train.T, y_train.T, params)
    print("final train score-", train_score)
    print("final validation score-", val_score)
    print("final test score-", test_score)
    print(costs)


def main():
    global x_train, y_train, x_test, y_test
    (x_train, y_train), (x_test, y_test) = load_data()
    (x_train, y_train), (NeuralNet.x_val, NeuralNet.y_val) = get_shuffle_validation(x_train,
                                                                                    y_train)  # Do the split but before x_transform, memory consumption
    y_train = NeuralNet.one_hot(y_train, NUM_OF_CLASSES)
    y_test = NeuralNet.one_hot(y_test, NUM_OF_CLASSES)
    NeuralNet.y_val = NeuralNet.one_hot(NeuralNet.y_val, NUM_OF_CLASSES)
    x_train = NeuralNet.transform_x(x_train)
    x_test = NeuralNet.transform_x(x_test)
    NeuralNet.x_val = NeuralNet.transform_x(NeuralNet.x_val)
    image_size = x_test.shape[1]
    layers_dims = [image_size, 20, 7, 5, 10]

    params, costs = NeuralNet.L_layer_model(x_train.T, y_train.T, layers_dims, num_iterations=1000, batch_size=256,
                                            learning_rate=0.009)
    print_scores(params, costs)

    # NeuralNet.use_batch_norm = True
    # params, costs = NeuralNet.L_layer_model(x_train.T, y_train.T, layers_dims, num_iterations=1000, batch_size=256,
    #                                         learning_rate=0.009)
    print_scores(params, costs)


if __name__ == "__main__":
    main()

    # test_size = y_test.shape[0]
    # image_size = x_test.shape[1] * x_test.shape[2]
    # xy_test = np.concatenate((x_test.reshape(test_size, -1), y_test.reshape((test_size, 1))), axis=1)  # [x|y]
    # np.random.shuffle(xy_test)
    # xy_validation = xy_test[:test_size // 5] # use 20% for validation
    # xy_test = xy_test[test_size // 5:]
    # y_validation = xy_validation[:, image_size]
    # y_test = xy_test[:, image_size]
    # x_validation = np.delete(xy_validation, image_size, axis=1)
    # x_test = np.delete(xy_test, image_size, axis=1)
    # x_train.reshape(test_size, -1)
    # return (transform_x(x_train), transform_y(y_train)), \
    #        (transform_x(x_validation), transform_y(y_validation)), \
    #        (transform_x(x_test), transform_y(y_test))

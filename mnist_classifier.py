import numpy as np
from tensorflow import keras
from NeuralNetwork import *

np.random.seed(101)

def transform_y(y_array):
    result = np.zeros((10, y_array.size), dtype=np.int8)
    for index in range(y_array.size):
        result[:, index][y_array[index]] = 1
    return result

def transform_x(x_array):
    return x_array / 255

def load_data():
    (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
    test_size = y_test.shape[0]
    image_size = x_test.shape[1] * x_test.shape[2]
    xy_test = np.concatenate((x_test.reshape(test_size, -1), y_test.reshape((test_size, 1))), axis=1)  # [x|y]
    np.random.shuffle(xy_test)
    xy_validation = xy_test[:test_size // 5] # use 20% for validation
    xy_test = xy_test[test_size // 5:]
    y_validation = xy_validation[:, image_size]
    y_test = xy_test[:, image_size]
    x_validation = np.delete(xy_validation, image_size, axis=1)
    x_test = np.delete(xy_test, image_size, axis=1)
    x_train.reshape(test_size, -1)
    return (transform_x(x_train), transform_y(y_train)), \
           (transform_x(x_validation), transform_y(y_validation)), \
           (transform_x(x_test), transform_y(y_test))


def main():
    (x_train, y_train), (x_validation, y_validation), (x_test, y_test) = load_data()
    image_size = x_test.shape[1]
    layers_dims = [image_size, 20, 7, 5, 10]
    params, costs = L_layer_model(x_validation.T, y_validation, layers_dims, num_iterations=1000, batch_size=100, learning_rate=0.009)
    print(costs)

if __name__ == "__main__":
    main()
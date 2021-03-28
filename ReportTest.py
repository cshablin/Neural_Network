import unittest
from NeuralNet import *
from mnist_classifier import *
from tensorflow import keras
np.random.seed(101)

class ReportTests(unittest.TestCase):
    # we use same test train sets for all 3 runs
    (x_train, y_train), (x_test, y_test) = load_data()
    y_train = one_hot(y_train, 10)
    y_test = one_hot(y_test, 10)

    def test_neural_net_regular(self):

        image_size = self.x_test.shape[1]
        layers_dims = [image_size, 20, 7, 5, 10]
        params, costs, final_validation_set = L_layer_model(self.x_train.T, self.y_train.T, layers_dims, num_iterations=1000, batch_size=64, learning_rate=0.009)
        test_score = predict(self.x_test.T, self.y_test.T, params)
        validation_score = predict(final_validation_set[0], final_validation_set[1], params)
        train_score = predict(self.x_train.T, self.y_train.T, params)
        print("final train score-", train_score)
        print("final validation score-", validation_score)
        print("final test score-", test_score)

    def test_neural_net_batch_norm(self):
        image_size = self.x_test.shape[1]
        layers_dims = [image_size, 20, 7, 5, 10]
        params, costs, final_validation_set = L_layer_model(self.x_train.T, self.y_train.T, layers_dims, num_iterations=1000, batch_size=32, learning_rate=0.009)
        test_score = predict(self.x_test.T, self.y_test.T, params)
        validation_score = predict(final_validation_set[0], final_validation_set[1], params)
        train_score = predict(self.x_train.T, self.y_train.T, params)
        print("final train score-", train_score)
        print("final validation score-", validation_score)
        print("final test score-", test_score)

    def test_neural_net_dropout(self):
        pass

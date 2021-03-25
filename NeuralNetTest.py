import unittest
from mnist_classifier import *

np.random.seed(101)


class NeuralNetTests(unittest.TestCase):

    def test_init_layers(self):
        layers_dims = [2, 3, 4]
        layers_params = initialize_parameters(layers_dims)
        last_layer = layers_params[2]
        self.assertEqual((4, 3), last_layer.weights.shape)
        # TODO: weight values are between -1 and 1
        # TODO: bias values are 0

    def test_net_forward_pass(self):
        layers_dims = [2, 3, 4]
        params = initialize_parameters(layers_dims)
        x = np.array([[5.0, 3.0],
                      [1.0, 4.0]])
        al, caches = linear_model_forward(x, params, use_batchnorm=False)
        self.assertEqual((4, 2), al.shape)

    def test_linear_forward(self):
        w = np.array([[5.0, 1.0, 3.0],
                      [1.0, 2.0, 1.0]])

        bias = np.array([[7.0],
                         [1.0]])

        activations = np.array([1.0, 1.0, 1.0])

        a, cache = linear_forward(activations, w, bias)
        self.assertTrue(np.array_equal(a, np.array([16.0, 5.0])))

    def test_soft_max(self):
        z = np.array([[2.0, 2.0, 2.0, 2.0]])
        a, activation_cache = soft_max(z.reshape(4, 1))
        self.assertTrue(np.array_equal(a, np.array([0.25, 0.25, 0.25, 0.25]).reshape(4, 1)))

    def test_relu(self):
        z = np.array([-2.0, 2.0, -2.0, 2.0])
        a, activation_cache = relu(z)
        self.assertTrue(np.array_equal(a, np.array([0, 2.0, 0, 2.0])))

    def test_compute_cost(self):
        predictions = np.array([[0.25,0.25,0.25,0.25],
                                [0.01,0.01,0.01,0.96]])
        targets = np.array([[0,0,0,1],
                            [0,0,0,1]])

        res = compute_cost(predictions.T, targets.T)
        self.assertTrue(np.isclose(0.713558177, res))

    def test_batch_norm(self):
        a = np.array([2,4,6,3,1,5]).reshape((2,3))
        na = apply_batchnorm(a)
        self.assertAlmostEqual(na[0][0],-0.866, delta=1e-4) # -0.866 = (2-4)/sqrt((4+4+8)/3)

    def test_linear_backward(self):
        w = np.array([[5.0, 1.0, 3.0],
                      [1.0, 2.0, 1.0]])
        bias = np.array([[7.0],
                         [1.0]])
        activations = np.array([[ 0.30532145,  0.66373246], [-5.37221442, -5.45698139], [ 2.52056354,  3.78283679]])

        z, linear_cache = linear_forward(activations, w, bias)
        dz = np.array([[2.0, 0.5], [1.0, 1.5]])
        dA_prev, dW, db = linear_backward(dz, linear_cache)
        self.assertEqual(activations.shape, dA_prev.shape)
        self.assertEqual(db.shape, bias.shape)
        self.assertEqual(dW.shape, w.shape)

    def test_linear_activation_backward(self):
        layers_dims = [2, 3, 4]
        params = initialize_parameters(layers_dims)
        # 4 sample batch !!!
        x = np.array([[5.0, 3.0, 2, 2],
                      [1.0, 4.0, 1, 3]])
        al, caches = linear_model_forward(x, params)
        layer_1_cache = caches[1]
        da = np.array([[2.0, 0.5, 2, 2], [1.0, 1.5, 2, 2], [1.0, 2.5, 2, 2]])
        res = linear_activation_backward(da, layer_1_cache, params[1].activation_func)

    def test_linear_model_backward(self):
        layers_dims = [2, 3, 4]
        params = initialize_parameters(layers_dims)
        # 4 sample batch !!!
        x = np.array([[5.0, 3.0, 2, 2],
                      [1.0, 4.0, 1, 3]])
        al, caches = linear_model_forward(x, params)
        y = np.eye(4)
        res = linear_model_backward(al, y, caches)
        res['dW1'].shape = (3,2)
        res['dW2'].shape = (4,3)

    def test_layer_model(self):
        layers_dims = [4, 3, 3, 4]
        X = np.array(([[5, 1, 2, 1],
                       [3, 1, 3, 5],
                       [2, 5, 3, 3],
                       [2, 3, 5, 1],
                       [1, 2, 2, 5],
                       [5, 1, 1, 1]])).T # reason for transpose: each sample must be a column
        Y = np.array([[1, 0, 0, 0, 0, 1],
                      [0, 0, 1, 0, 0, 0],
                      [0, 0, 0, 1, 0, 0],
                      [0, 1, 0, 0, 1, 0]])
        params, costs = L_layer_model(X, Y, layers_dims, num_iterations=1000, batch_size=1, learning_rate=0.001)
        self.assertLess(costs[len(costs) - 1], 0.02)
        # self.assertEqual(len(costs), 9)

    def test_predict(self):
        layers_dims = [4, 2, 3, 4]
        X = np.array(([[5, 1, 2, 1],
                       [3, 1, 3, 5],
                       [2, 5, 3, 3],
                       [2, 3, 5, 1],
                       [1, 2, 2, 5],
                       [5, 1, 1, 1]])).T  # reason for transpose: each sample must be a column
        Y = np.array([[1, 0, 0, 0, 0, 1],
                      [0, 0, 1, 0, 0, 0],
                      [0, 0, 0, 1, 0, 0],
                      [0, 1, 0, 0, 1, 0]])
        params, costs = L_layer_model(X, Y, layers_dims, num_iterations=1000, batch_size=1, learning_rate=0.009)
        X_test = np.array([[5, 2, 1, 1]]).T
        Y_test = np.array([[1, 0, 0, 0]]).T
        accuracy = predict(X, Y, params)
        self.assertGreater(accuracy, 0.8)
        accuracy = predict(X_test, Y_test, params)
        # self.assertGreater(accuracy, 0.3)


class ReportTests(unittest.TestCase):
    # we use same test train sets for all 3 runs
    (x_train, y_train), (x_test, y_test) = load_data()
    y_train = one_hot(y_train, 10)
    y_test = one_hot(y_test, 10)

    def test_neural_net_regular(self):

        image_size = self.x_test.shape[1]
        layers_dims = [image_size, 20, 7, 5, 10]
        params, costs = L_layer_model(self.x_train.T, self.y_train.T, layers_dims, num_iterations=1000, batch_size=64, learning_rate=0.009)
        test_score = predict(self.x_test.T, self.y_test.T, params)
        train_score = predict(self.x_train.T, self.y_train.T, params)
        print("final train score-", train_score)
        print("final validation score-", test_score)
        print("final test score-", test_score)

    def test_neural_net_batch_norm(self):
        pass

    def test_neural_net_dropout(self):
        pass

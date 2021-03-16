import unittest
import numpy as np

from NeuralNetwork import Network, Layer


class NeuralNetworkTests(unittest.TestCase):

    def test_init_layers(self):

        network = Network()
        layers_dims = [2, 3, 4]
        layers_params = network.initialize_parameters(layers_dims)
        last_layer = layers_params[2]
        self.assertEqual((4, 3), last_layer.weights.shape)

    def test_net_forward_pass(self):

        network = Network()
        layers_dims = [2, 3, 4]
        network.initialize_parameters(layers_dims)
        x = np.array([[5.0, 3.0],
                      [1.0, 4.0]])
        result = network.linear_model_forward(x)
        self.assertEqual((4, 2), result.shape)

    def test_linear_forward(self):
        layer = Layer(None, None)
        w = np.array([[5.0, 1.0, 3.0],
                      [1.0, 2.0, 1.0]])

        bias = np.array([[7.0],
                         [1.0]])

        activations = np.array([1.0, 1.0, 1.0])

        result = layer.linear_forward(activations, w, bias)
        self.assertTrue(np.array_equal(result, np.array([16.0, 5.0])))

    def test_soft_max(self):
        layer = Layer(None, None)
        z = np.array([[2.0, 2.0, 2.0, 2.0]])
        result = layer.soft_max(z.reshape(1, 4))
        self.assertTrue(np.array_equal(result, np.array([0.25, 0.25, 0.25, 0.25]).reshape(1, 4)))

    def test_relu(self):
        layer = Layer(None, None)
        z = np.array([-2.0, 2.0, -2.0, 2.0])
        result = layer.relu(z)
        self.assertTrue(np.array_equal(result, np.array([0, 2.0, 0, 2.0])))

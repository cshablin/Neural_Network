import unittest
import numpy as np
from NeuralNetwork import Network, Layer

np.random.seed(101)


class NeuralNetworkTests(unittest.TestCase):

    def test_init_layers(self):

        network = Network()
        layers_dims = [2, 3, 4]
        layers_params = network.initialize_parameters(layers_dims)
        last_layer = layers_params[2]
        self.assertEqual((4, 3), last_layer.weights.shape)
        # TODO: weight values are between -1 and 1
        # TODO: bias values are 0

    def test_net_forward_pass(self):

        network = Network()
        layers_dims = [2, 3, 4]
        network.initialize_parameters(layers_dims)
        x = np.array([[5.0, 3.0],
                      [1.0, 4.0]])
        result = network.linear_model_forward(x, use_batchnorm=True)
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
        result = layer.soft_max(z.reshape(4, 1))
        self.assertTrue(np.array_equal(result, np.array([0.25, 0.25, 0.25, 0.25]).reshape(4, 1)))

    def test_relu(self):
        layer = Layer(None, None)
        z = np.array([-2.0, 2.0, -2.0, 2.0])
        result = layer.relu(z)
        self.assertTrue(np.array_equal(result, np.array([0, 2.0, 0, 2.0])))

    def test_compute_cost(self):
        predictions = np.array([[0.25,0.25,0.25,0.25],
                                [0.01,0.01,0.01,0.96]])
        targets = np.array([[0,0,0,1],
                            [0,0,0,1]])

        layer = Layer(None, None)
        res = layer.compute_cost(predictions.T, targets.T)
        self.assertTrue(np.isclose(0.713558177, res))

    def test_batch_norm(self):
        a = np.array([2,4,6,3,1,5]).reshape((2,3))
        layer = Layer(None, None)
        na = layer.apply_batchnorm(a)
        self.assertAlmostEqual(na[0][0],-0.866, delta=1e-4) # -0.866 = (2-4)/sqrt((4+4+8)/3)

    def test_linear_backward(self):
        layer = Layer(None, None)
        w = np.array([[5.0, 1.0, 3.0],
                      [1.0, 2.0, 1.0]])
        bias = np.array([[7.0],
                         [1.0]])
        activations = np.array([[ 0.30532145,  0.66373246], [-5.37221442, -5.45698139], [ 2.52056354,  3.78283679]])

        layer.linear_forward(activations, w, bias)
        dz = np.array([[2.0, 0.5], [1.0, 1.5]])
        dA_prev, dW, db = layer.linear_backward(dz)
        self.assertEqual(activations.shape, dA_prev.shape)
        self.assertEqual(db.shape, bias.shape)
        self.assertEqual(dW.shape, w.shape)

    def test_linear_activation_backward(self):
        network = Network()
        layers_dims = [2, 3, 4]
        network.initialize_parameters(layers_dims)
        x = np.array([[5.0, 3.0],
                      [1.0, 4.0]])
        result = network.linear_model_forward(x)
        layer_1 = network.index_2_layer[1]
        da = np.array([[2.0, 0.5], [1.0, 1.5], [1.0, 2.5]])
        res = layer_1.linear_activation_backward(da)

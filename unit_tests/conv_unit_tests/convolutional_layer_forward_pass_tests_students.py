import unittest
import numpy as np
import pickle
import torch
from torch.nn import Conv2d, MaxPool2d, AvgPool2d
from torch import Tensor
from torch.autograd import Variable

from netlib.CNN import get_indices, Convolution, Pooling
from netlib.Layer import InitType
np.random.seed(10)


class TestCalculator(unittest.TestCase):
    def setUp(self):
        with open('input_tensor.pickle', 'rb') as file:
            self.input = pickle.load(file)

    def check_index(self, kernel_size, pad, stride, g_k, g_i, g_j):
        if kernel_size == 2 and stride == 2:
            print(kernel_size)
        k, i, j = get_indices(self.input.shape, kernel_size, kernel_size, padding=pad, stride=stride)
        self.assertEqual(np.array_equal(k, g_k), True)
        self.assertEqual(np.array_equal(i, g_i), True)
        self.assertEqual(np.array_equal(j, g_j), True)

    def test_convolutional_layers(self):
        with open('convolutional_results.pickle', 'rb') as file:
            indices, weights, outputs = pickle.load(file)
        iter = 0
        for kernel_size in range(1, 6):
            for zero_pad in range(0, 2):
                for stride in range(1, 3):
                    if (len(self.input[0][0]) + 2 * zero_pad - kernel_size) % stride != 0:
                        continue
                    with self.subTest(kernel_size=kernel_size, zero_pad=zero_pad, stride=stride):
                        k, i, j = indices[iter]
                        self.check_index(kernel_size=kernel_size, pad=zero_pad, stride=stride, g_k=k, g_i=i, g_j=j)

                        # forward pass
                        layer = Convolution(kernel_size, 64, 3, zero_pad, stride, False, InitType.HeNormal)
                        self.assertEqual(np.all(layer.W.shape == weights[iter].shape), True)
                        layer.W = weights[iter]
                        output = outputs[iter]
                        self.assertAlmostEqual(np.max(np.abs(layer(self.input, phase='eval') - output)), 0)

                        # backward pass
                        for use_bias in [True, False]:
                            my_layer = Convolution(kernel_size, 64, 3, zero_pad, stride, use_bias, InitType.HeNormal)
                            my_layer_forward_pass_result = my_layer(self.input)
                            dx, dw = my_layer.backward(self.input, np.ones(my_layer_forward_pass_result.shape))
                            if use_bias:
                                dw, db = dw
                            else:
                                db = None
                            torch_input = Variable(Tensor(self.input), requires_grad=True)
                            torch_layer = Conv2d(in_channels=3, out_channels=64, kernel_size=kernel_size, stride=stride,
                                                 padding=zero_pad, bias=use_bias)
                            torch_layer.weight = torch.nn.Parameter(Tensor(my_layer.W))
                            if use_bias:
                                torch_layer.bias = torch.nn.Parameter(Tensor(my_layer.b))
                            torch_layer_forward_pass_result = torch_layer(torch_input)
                            torch_layer_forward_pass_result.backward(torch.ones(my_layer_forward_pass_result.shape))
                            torch_grad_input, torch_grad_weight = torch_input.grad, torch_layer.weight.grad
                            assert np.allclose(dx, torch_grad_input.numpy(), atol=1e-2)
                            assert np.allclose(dw, torch_grad_weight.numpy(), atol=1e-2)
                            if use_bias:
                                torch_grad_bias = torch_layer.bias.grad
                                assert np.allclose(db, torch_grad_bias.numpy(), atol=1e-2)
                        iter += 1

    def test_pooling_layers(self):
        with open('pooling_results.pickle', 'rb') as file:
            indices, outputs = pickle.load(file)
        iter = 0
        for kernel_size in range(2, 4):
            for stride in range(1, 4):
                for type in ['Max', 'Avg']:
                    if (len(self.input[0][0]) - kernel_size) % stride != 0:
                        continue
                    with self.subTest(kernel_size=kernel_size, type=type, stride=stride):
                        k, i, j = indices[iter]
                        self.check_index(kernel_size=kernel_size, pad=0, stride=stride, g_k=k, g_i=i, g_j=j)

                        # forward pass
                        my_layer = Pooling(kernel_size, stride, type)
                        my_layer_forward_pass_result = my_layer(self.input, phase='eval')
                        output = outputs[iter]
                        self.assertAlmostEqual(np.max(np.abs(my_layer_forward_pass_result - output)), 0)

                        # backward pass
                        dx, _ = my_layer.backward(self.input, np.ones(my_layer_forward_pass_result.shape))
                        torch_input = Variable(Tensor(self.input), requires_grad=True)
                        if type == 'Max':
                            torch_layer = MaxPool2d(kernel_size=kernel_size, stride=stride)
                        elif type == 'Avg':
                            torch_layer = AvgPool2d(kernel_size=kernel_size, stride=stride)
                        else:
                            raise Exception
                        torch_layer_forward_pass_result = torch_layer(torch_input)
                        torch_layer_forward_pass_result.backward(torch.ones(my_layer_forward_pass_result.shape))
                        torch_grad_input = torch_input.grad
                        assert np.allclose(dx, torch_grad_input.numpy(), atol=1e-4)

                        iter += 1


if __name__ == "__main__":
    unittest.main()

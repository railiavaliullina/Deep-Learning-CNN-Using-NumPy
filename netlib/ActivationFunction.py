from abc import abstractmethod
import numpy as np
from enum import Enum


class ActivationFunction(Enum):
    ReLU = 0
    Sigmoid = 1
    Tanh = 2
    Linear = 3


class BaseActivationFunctionClass(object):
    @abstractmethod
    def __call__(self, x, phase):
        return x

    @property
    def trainable(self):
        return True

    def get_grad(self):
        pass

    @abstractmethod
    def backward(self, layer_input, dy):
        pass


class ReLU(BaseActivationFunctionClass):
    def __call__(self, x, phase=None):
        return np.maximum(0, x)

    def backward(self, layer_input, dy):
        d_r = layer_input > 0
        return dy * d_r, None


class Sigmoid(BaseActivationFunctionClass):
    def __call__(self, x, phase=None):
        return 1/(1 + np.exp(-x))

    def backward(self, layer_input, dy):
        sigmoid_ = self.__call__(layer_input)
        return dy * sigmoid_ * (1 - sigmoid_), None


class Tanh(BaseActivationFunctionClass):
    def __call__(self, x, phase=None):
        return (np.exp(2*x) - 1)/(np.exp(2*x) + 1)

    def backward(self, layer_input, dy):
        tanh_ = self.__call__(layer_input)
        return dy * (1 - tanh_ ** 2), None


class LinearA(BaseActivationFunctionClass):

    def __call__(self, x, phase=None):
        return x

    def backward(self, layer_input, dy):
        return dy, None

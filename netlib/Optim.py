import numpy as np
from enum import Enum


class OptimType(Enum):
    SGD = 0
    MomentumSGD = 1


class BaseOptim(object):
    def __init__(self, optim_type, learning_rate, momentum, net):
        self.optim_type = optim_type
        self.learning_rate = learning_rate
        self.net = net
        # self.loss = loss
        self.momentum = momentum
        self.v = []
        for layer_name, layer in self.net.layers_dict.items():
            if layer_name.startswith('fc'):
                v = (np.zeros(layer.W.shape), np.zeros(layer.b.shape)) if layer.use_bias else np.zeros(layer.W.shape)
            else:
                v = None
            self.v.append(v)

    def update_rule(self, dW, i):
        if self.optim_type.name == 'SGD':
            if isinstance(dW, tuple):
                term = (- self.learning_rate * dW[0], - self.learning_rate * dW[1])
            else:
                term = - self.learning_rate * dW
        else:
            if isinstance(dW, tuple):
                w_term = self.momentum * self.v[i][0] - self.learning_rate * dW[0]
                if self.v[i][1] is not None:
                    b_term = self.momentum * self.v[i][1] - self.learning_rate * dW[1]
                    term = (w_term, b_term)
                else:
                    term = w_term

            else:
                v = self.v[i][0] if isinstance(self.v[i], tuple) else self.v[i]  # при инициализации
                term = self.momentum * v - self.learning_rate * dW
            self.v[i] = term
        return term

    def minimize(self, dl_dz, layers_inputs):
        # backward pass
        current_grad = dl_dz
        reversed_layer_names, reversed_layers = list(self.net.layers_dict.keys())[::-1], list(self.net.layers_dict.values())[::-1]
        reversed_layers_ids = np.arange(len(reversed_layer_names))[::-1]
        for i, reversed_layer_name, reversed_layer in zip(reversed_layers_ids, reversed_layer_names, reversed_layers):
            if reversed_layer_name.startswith('pool'):
                current_grad = current_grad.reshape(layers_inputs[i].shape[0], layers_inputs[i].shape[1], layers_inputs[i].shape[2]//2,
                                                    layers_inputs[i].shape[3]//2)
            # current_grad = current_grad.reshape()
            current_grad, dW = reversed_layer.backward(layers_inputs[i], current_grad)
            if reversed_layer_name.startswith('fc') or reversed_layer_name.startswith('conv'):
                reversed_layer.update_weights(self.update_rule(dW, i))
        return current_grad

from abc import abstractmethod
import numpy as np
from enum import Enum

np.random.seed(0)


class InitType(Enum):
    Normal = 0
    Uniform = 1
    HeNormal = 2
    HeUniform = 3
    XavierNormal = 4
    XavierUniform = 5
    GlorotNormal = 6
    GlorotUniform = 7


class BaseLayerClass(object):
    @abstractmethod
    def __call__(self, x, phase):
        return x

    @property
    def trainable(self):
        return False

    def get_grad(self):
        pass

    @abstractmethod
    def backward(self, layer_input, dy):
        pass

    def update_weights(self, update_term):
        pass

    def get_nrof_trainable_params(self):
        return 0


class Linear(BaseLayerClass):
    def __init__(self, input_shape, output_shape, use_bias, initialization_type, regularization_type, weight_decay,
                 dtype=np.float32, layer_name='', cfg=None):
        self.input_shape = input_shape
        self.output_shape = output_shape
        self.use_bias = use_bias
        self.initialization_type = initialization_type
        self.regularization_type = regularization_type
        self.weight_decay = weight_decay
        self.dtype = dtype
        self.layer_name = layer_name
        self.cfg = cfg
        self.b = None
        scale, l = None, None

        if self.initialization_type.name.endswith('Normal'):
            if self.initialization_type.name == 'Normal':
                scale = self.cfg['model'][self.layer_name]['scale']
            elif self.initialization_type.name == 'HeNormal':
                scale = np.sqrt(2 / self.output_shape)
            elif self.initialization_type.name == 'XavierNormal':
                scale = np.sqrt(1 / self.output_shape)
            elif self.initialization_type.name == 'GlorotNormal':
                scale = np.sqrt(1 / (self.output_shape + self.input_shape))
            self.W = np.random.normal(size=[self.input_shape, self.output_shape], scale=scale)
            if self.use_bias:
                self.b = np.random.normal(size=[self.output_shape], scale=scale)
        else:
            if self.initialization_type.name == 'Uniform':
                l = self.cfg['model'][self.layer_name]['l']
            elif self.initialization_type.name == 'HeUniform':
                l = np.sqrt(6 / self.output_shape)
            elif self.initialization_type.name == 'XavierUniform':
                l = np.sqrt(3 / self.output_shape)
            elif self.initialization_type.name == 'GlorotUniform':
                l = np.sqrt(6 / (self.output_shape + self.input_shape))
            self.W = np.random.uniform(-l, l, size=[self.input_shape, self.output_shape])
            if self.use_bias:
                self.b = np.random.uniform(-l, l, size=[self.output_shape])

    def __call__(self, x, phase='train'):
        # forward pass
        if len(x.shape) > 3:
            x = x.reshape((x.shape[0], -1))
        y = np.dot(x, self.W)
        if self.use_bias:
            y = y + self.b
        return y

    @property
    def trainable(self):
        return True

    def backward(self, layer_input, dy):
        if len(layer_input.shape) > 3:
            layer_input = layer_input.reshape(layer_input.shape[0], -1)
        dW = np.dot(layer_input.T, dy)
        assert dW.shape == self.W.shape
        dW_ = dW
        if self.use_bias:
            db = dy.mean(axis=0) * layer_input.shape[0]
            assert db.shape == self.b.shape
            dW_ = (dW, db)
        out_grad = np.dot(dy, self.W.T)
        return out_grad, dW_

    def update_weights(self, update_term):
        if isinstance(update_term, tuple):
            self.W = self.W + update_term[0]
            if self.use_bias:
                self.b = self.b + update_term[1]
        else:
            self.W = self.W + update_term
            if self.use_bias:
                self.b = self.b + update_term

    def get_nrof_trainable_params(self):
        nrof_params = self.input_shape * self.output_shape
        if self.use_bias:
            nrof_params += len(self.b)
        print(f'{self.layer_name} trainable params number:', nrof_params)
        return nrof_params

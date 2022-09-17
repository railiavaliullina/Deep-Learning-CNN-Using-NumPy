import numpy as np
import pickle

from netlib.Layer import Linear
from netlib.CNN import Convolution, Pooling
from netlib.ActivationFunction import LinearA, Sigmoid, ReLU, Tanh
from netlib.Loss import CrossEntropyLoss
from netlib.Optim import BaseOptim


class BaseModelClass(object):
    def __init__(self, cfg):
        # инициализация слоев на основе конфига
        self.cfg = cfg
        self.layers_dict = {}
        self.net_nrof_trainable_params = 0  # количество обучаемых параметров всей сети
        self._train = True
        k = 1

        for layer in cfg['model']['layers']:
            if isinstance(layer, str):
                if layer.startswith('fc'):
                    layer_ = Linear(input_shape=cfg['model'][layer]['layer_input_dim'],
                                    output_shape=cfg['model'][layer]['layer_output_dim'],
                                    use_bias=cfg['model'][layer]['use_bias'],
                                    initialization_type=cfg['model'][layer]['init_type'],
                                    regularization_type='',
                                    weight_decay='',
                                    layer_name=layer,
                                    cfg=cfg)

                    # layer_nrof_params = layer_.get_nrof_trainable_params()
                    # self.net_nrof_trainable_params += layer_nrof_params
                    # self.layers_dict[layer] = layer_
                elif layer.startswith('conv'):
                    layer_ = Convolution(kernel_size=cfg['model'][layer]['kernel_size'],
                                         nrof_filters=cfg['model'][layer]['nrof_filters'],
                                         kernel_depth=cfg['model'][layer]['kernel_depth'],
                                         zero_pad=cfg['model'][layer]['zero_pad'],
                                         stride=cfg['model'][layer]['stride'],
                                         use_bias=cfg['model'][layer]['use_bias'],
                                         initialization_type=cfg['model'][layer]['initialization_type'])

                elif layer.startswith('pool'):
                    layer_ = Pooling(kernel_size=cfg['model'][layer]['kernel_size'],
                                     stride=cfg['model'][layer]['stride'])
                else:
                    raise Exception

                self.layers_dict[layer] = layer_

            else:
                if layer.name == "Linear":
                    activation_func = LinearA()
                elif layer.name == "Sigmoid":
                    activation_func = Sigmoid()
                elif layer.name == "ReLU":
                    activation_func = ReLU()
                elif layer.name == "Tanh":
                    activation_func = Tanh()
                else:
                    raise Exception

                self.layers_dict[layer.name + f'_{k}'] = activation_func
                k += 1
        # вывод количества обучаемых параметров всей сети
        # self.get_nrof_trainable_params()
        # инициализация лосса
        self.loss = CrossEntropyLoss()
        # инициализация оптимайзера
        self.opt = BaseOptim(optim_type=cfg['train']['optim_type'], learning_rate=cfg['train']['learning_rate'],
                             momentum=cfg['train']['momentum'], net=self)

    def make_step(self, input, labels):
        # calling forward pass
        activations = self.__call__(input)
        logits = activations[-1]
        loss_on_batch = np.mean(self.loss(logits, labels))
        # loss grad
        dl_dz = self.loss.backward(logits, labels)
        # calling backward pass
        dl_dx = self.opt.minimize(dl_dz, [input] + activations[:-1])
        return loss_on_batch

    def __call__(self, input):
        # forward pass
        layers_activations = []
        for layer_name, layer in self.layers_dict.items():
            input = layer(input)
            layers_activations.append(input)
        assert len(layers_activations) == len(self.layers_dict)
        return layers_activations

    @property
    def train(self):
        return self._train

    @train.setter
    def train(self, value):
        self._train = value

    def dump_model(self, path):
        with open(path + '.pickle', 'wb') as f:
            pickle.dump(self.layers_dict, f)

    def load_weights(self, path):
        with open(path + '.pickle', 'rb') as f:
            self.layers_dict = pickle.load(f)

    def get_nrof_trainable_params(self):
        print(f'\nNet trainable params number: {self.net_nrof_trainable_params}\n')

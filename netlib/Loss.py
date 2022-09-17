import numpy as np


class BaseLossClass(object):
    def __init__(self):
        pass

    def __call__(self, logits, labels, phase='eval'):
        pass

    def get_grad(self):
        pass

    def backward(self, logits, labels):
        pass


class CrossEntropyLoss(BaseLossClass):

    def __call__(self, logits, labels, phase='eval'):
        return - np.sum(logits * labels, 1) + np.log(np.sum(np.exp(logits), axis=-1))

    def backward(self, logits, labels):
        b_size = logits.shape[0]
        dl = - labels + np.exp(logits) / np.exp(logits).sum(axis=-1, keepdims=True)
        return dl / b_size

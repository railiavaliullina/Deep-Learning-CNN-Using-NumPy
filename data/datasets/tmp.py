import numpy as np
from collections import Counter
import matplotlib.pyplot as plt
import multiprocessing

from configs.config import cfg
from configs.mnist_config import cfg as mnist_cfg
from configs.cifar_config import cfg as cifar_cfg
from data.dataloaders.Transform import *


class BaseDataset(object):
    transforms_ = None
    transforms = None
    show = None
    # dataset_cfg = None #cifar_cfg if cfg["dataset"] == 'cifar' else mnist_cfg

    def __init__(self, data_path, dataset_type, transforms, nrof_classes):
        """
        :param data_path (string): путь до файла с данными.
        :param dataset_type (string): (['train', 'valid', 'test']).
        :param transforms (list): список необходимых преобразований изображений.
        :param nrof_classes (int): количество классов в датасете.
        """
        self.dataset_cfg = cifar_cfg if cfg["dataset"] == 'cifar' else mnist_cfg
        self.data_path = data_path
        self.dataset_type = dataset_type
        self.transforms = transforms
        self.nrof_classes = nrof_classes
        self.images, self.labels = [], []
        BaseDataset.transforms = self.dataset_cfg["transform_parameters"][self.dataset_type]
        BaseDataset.transforms_ = Transforms(self.dataset_type, self.dataset_cfg)
        BaseDataset.show = self.dataset_cfg["transform_parameters"]["show_at_every_transform"]
        self.pool = multiprocessing.Pool()

    def __len__(self):
        """
        :return: размер выборки
        """
        return len(self.labels)

    def one_hot_labels(self, label):
        """
        :param label: метка класса
        :return: one-hot encoding вектор
        """
        return np.squeeze(np.eye(self.nrof_classes)[label.reshape(-1)])

    @staticmethod
    def apply_augmentation(img):
        images_to_plot = []
        for transform in BaseDataset.transforms:
            to_apply = np.random.choice([True, False], 1)[0]  # достать из конфига
            # if to_apply:
            image = getattr(BaseDataset.transforms_, transform)(image=img, show=BaseDataset.show)
            images_to_plot.append(image)
        return img

    @staticmethod
    def tmp(value):
        value = Transforms.GaussianBlur(value, 5)
        # image = getattr(BaseDataset.transforms_, BaseDataset.transform[0])(image=value, show=BaseDataset.show)
        return value

    def __getitem__(self, idx):
        """
        :param idx: индекс элемента в выборке
        :return: preprocessed image and label
        """
        with self.pool:# self.images[idx]
            img = self.pool.map(BaseDataset.tmp, self.images[idx]) #np.array([self.apply_augmentation(self.images[i]) for i in idx])
        label = np.array([self.one_hot_labels(self.labels[i]) for i in idx])
        return img, label

    def show_statistics(self):
        """
        Необходимо вывести количество элементов в датасете, количество классов и количество элементов в каждом классе
        """
        # количество элементов в датасете
        nrof_elements = self.__len__()
        print(f'\nNumber of elements in {self.dataset_type} set: {nrof_elements}')
        # количество классов
        print(f'Number of classes in {self.dataset_type} set: {self.nrof_classes}')
        # количество элементов в каждом классе
        nrof_elements_in_each_class = dict(sorted(Counter(np.array(self.labels)).items()))
        self.nrof_elements_in_each_class = nrof_elements_in_each_class
        print(f'Number of elements in each class: {nrof_elements_in_each_class}')

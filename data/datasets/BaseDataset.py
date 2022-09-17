import numpy as np
from collections import Counter

from configs.mnist_config import cfg as mnist_cfg
from configs.cifar_config import cfg as cifar_cfg
from data.dataloaders.Transform import *
from utils.visualization_utils import show_batch


class BaseDataset(object):
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
        self.transforms = self.dataset_cfg["transform_parameters"][self.dataset_type]
        self.transforms_ = Transforms(self.dataset_type, self.dataset_cfg)
        self.show = self.dataset_cfg["transform_parameters"]["show_at_every_transform"]
        self.images_ = []

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

    def apply_augmentation(self, img):
        applied_transforms = []
        for i, transform in enumerate(self.transforms):
            probabilities = [self.dataset_cfg["transform_parameters"][self.dataset_type][transform]["p"],
                             1 - self.dataset_cfg["transform_parameters"][self.dataset_type][transform]["p"]]
            to_apply = np.random.choice([True, False], 1, p=probabilities)[0]
            if to_apply:
                # print(f'{i}, {transform} was applied')
                img = getattr(self.transforms_, transform)(image=np.asarray(img, dtype=np.float32), show=self.show)
                applied_transforms.append(transform)

        if self.dataset_cfg["save_image_augmentations"]:
            show_batch((self.transforms_.saved_images, self.transforms_.saved_labels), grid_size=(2, 7))
        return img, applied_transforms

    def __getitem__(self, idx):
        """
        :param idx: индекс элемента в выборке
        :return: preprocessed image and label
        """
        out = np.asarray(list(map(self.apply_augmentation, self.images[idx])))
        img, applied_transforms = out[:, 0], out[:, 1]
        label = np.asarray(list(map(self.one_hot_labels, self.labels[idx])))

        if self.dataset_cfg["save_batch_augmentations"]:
            show_batch((img, [self.labels[i] for i in idx]), grid_size=(4, 4))
        return img, label

    def show_statistics(self):
        """
        Необходимо вывести количество элементов в датасете, количество классов и количество элементов в каждом классе
        """
        # количество элементов в датасете
        nrof_elements = self.__len__()
        print(f'Number of elements in {self.dataset_type} set: {nrof_elements}')
        # количество классов
        print(f'Number of classes in {self.dataset_type} set: {self.nrof_classes}')
        # количество элементов в каждом классе
        nrof_elements_in_each_class = dict(sorted(Counter(np.array(self.labels)).items()))
        self.nrof_elements_in_each_class = nrof_elements_in_each_class
        print(f'Number of elements in each class: {nrof_elements_in_each_class}\n')

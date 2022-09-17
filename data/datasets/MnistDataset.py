import pickle
import numpy as np
import gzip
from data.datasets.BaseDataset import BaseDataset
from configs.mnist_config import cfg


class MnistDataset(BaseDataset):
    def __init__(self, data_path, dataset_type, transforms, nrof_classes):
        super().__init__(data_path, dataset_type, transforms, nrof_classes)

        self.read_data()

    def read_data(self):
        """
        Считывание данных по заданному пути+вывод статистики.
        """
        # считывание данных по заданному пути
        with gzip.open(self.data_path, 'rb') as f:
            train_set, valid_set, test_set = pickle.load(f, encoding='latin1')

        if self.dataset_type == 'train':
            images, labels = np.concatenate([train_set[0], valid_set[0]]), np.concatenate([train_set[1], valid_set[1]])
            image_size = int(images.shape[1]/np.sqrt(images.shape[1]))
            self.images, self.labels = (np.reshape(images, (images.shape[0], image_size, image_size)), labels)
        else:
            images, labels = test_set
            image_size = int(images.shape[1] / np.sqrt(images.shape[1]))
            self.images, self.labels = (np.reshape(images, (images.shape[0], image_size, image_size)), labels)
        # вывод статистики
        self.show_statistics()

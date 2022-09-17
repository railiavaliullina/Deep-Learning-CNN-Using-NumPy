import numpy as np
import os
import pickle

from data.datasets.BaseDataset import BaseDataset
from configs.config import cfg


class CifarDataset(BaseDataset):
    def __init__(self, data_path, dataset_type, transforms, nrof_classes):
        super().__init__(data_path, dataset_type, transforms, nrof_classes)

        self.read_data()

    @staticmethod
    def load_pickle(f):
        return pickle.load(f, encoding='latin1')

    def load_batch(self, filename):
        with open(filename, 'rb') as f:
            d = self.load_pickle(f)
            images = d['data'].reshape(10000, 3, 32, 32)
            labels = d['labels']
            return images, labels

    def load_dataset(self):
        if self.dataset_type == 'train':
            for batch_number in range(1, 6):
                images, labels = self.load_batch(os.path.join(self.data_path, f'data_batch_{batch_number}'))
                self.images.extend(images)
                self.labels.extend(labels)
        else:
            self.images, self.labels = self.load_batch(self.data_path)
        self.images, self.labels = np.array(self.images), np.array(self.labels)

    def read_data(self):
        """
        Считывание данных по заданному пути+вывод статистики.
        """
        # считывание данных по заданному пути
        self.load_dataset()
        # self.images, self.labels = torch.tensor(self.images), torch.tensor(self.labels)

        # вывод статистики
        self.show_statistics()

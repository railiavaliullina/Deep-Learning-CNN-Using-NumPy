import numpy as np
from enum import Enum
from collections import Counter


class SampleType(Enum):
    DEFAULT = 0  # равномерно из всего датасета
    UPSAMPLE = 1  # равномерно из каждого класса, увеличивая размер каждого класса до максимального
    DOWNSAMPLE = 2  # равномерно из каждого класса, уменьшая размер каждого класса до минимального
    PROBABILITIES = 3  # случайно из каждого класса в зависимости от указанных вероятностей


class Sampler(object):
    def __init__(self, dataset, nrof_classes, epoch_size, shuffle, batch_size, probabilities=None, replace=False):
        self.dataset = dataset
        self.nrof_classes = nrof_classes
        self.shuffle = shuffle
        self.batch_size = batch_size
        self.epoch_size = epoch_size
        self.probabilities = probabilities
        self.replace = replace
        self.dataset_ids = np.arange(len(self.dataset))
        self.cursors = [1 for _ in range(self.nrof_classes)]
        assert self.epoch_size is not None, 'epoch_size parameter is None'

    def shuffle_dataset(self):
        np.random.shuffle(self.dataset_ids)
        self.dataset.images = self.dataset.images[self.dataset_ids]
        self.dataset.labels = self.dataset.labels[self.dataset_ids]

    def DEFAULT(self):
        if self.shuffle:
            self.shuffle_dataset()

        for _ in range(self.epoch_size):
            ind = np.random.choice(self.dataset_ids, self.batch_size, replace=self.replace)
            img, labels = self.dataset[ind]
            yield img, labels

    def UPSAMPLE(self):
        if self.shuffle:
            self.shuffle_dataset()

        class_with_max_nrof_elements = max(self.dataset.nrof_elements_in_each_class,
                                           key=self.dataset.nrof_elements_in_each_class.get)
        max_nrof_elements = self.dataset.nrof_elements_in_each_class[class_with_max_nrof_elements]
        for label, nrof_elements in self.dataset.nrof_elements_in_each_class.items():
            if label != class_with_max_nrof_elements:
                current_class_ids = np.where(np.array(self.dataset.labels) == label)[0]
                ind = np.random.choice(current_class_ids, max_nrof_elements - nrof_elements, replace=self.replace)
                images_to_add, labels_to_add = self.dataset.images[ind], self.dataset.labels[ind]
                self.dataset.images = np.concatenate((self.dataset.images, images_to_add))
                self.dataset.labels = np.concatenate((self.dataset.labels, labels_to_add))

        nrof_elements_in_each_class = dict(sorted(Counter(np.array(self.dataset.labels)).items())).values()
        assert np.all(
            np.ones(len(nrof_elements_in_each_class)) * max_nrof_elements == list(nrof_elements_in_each_class))

        for _ in range(self.epoch_size):
            ind = np.random.choice(self.dataset_ids, self.batch_size, replace=self.replace)
            yield self.dataset[ind]

    def DOWNSAMPLE(self):
        if self.shuffle:
            self.shuffle_dataset()

        class_with_min_nrof_elements = min(self.dataset.nrof_elements_in_each_class,
                                           key=self.dataset.nrof_elements_in_each_class.get)
        min_nrof_elements = self.dataset.nrof_elements_in_each_class[class_with_min_nrof_elements]
        images, labels = [], []
        for label, nrof_elements in self.dataset.nrof_elements_in_each_class.items():
            current_class_ids = np.where(np.array(self.dataset.labels) == label)[0]
            ind = np.random.choice(current_class_ids, min_nrof_elements, replace=self.replace)
            images.extend(self.dataset.images[ind])
            labels.extend(self.dataset.labels[ind])

        self.dataset.images = images
        self.dataset.labels = labels
        nrof_elements_in_each_class = dict(sorted(Counter(np.array(self.dataset.labels)).items())).values()
        assert np.all(
            np.ones(len(nrof_elements_in_each_class)) * min_nrof_elements == list(nrof_elements_in_each_class))

        self.dataset_ids = np.arange(len(self.dataset))
        for _ in range(self.epoch_size):
            ind = np.random.choice(self.dataset_ids, self.batch_size, replace=self.replace)
            yield self.dataset[ind]

    def PROBABILITIES(self):
        if self.shuffle:
            self.shuffle_dataset()
        for _ in range(self.epoch_size):
            cur_inds = np.array([self.dataset_ids[np.where(self.dataset.labels == label)[0]][:self.cursors[label]]
                                 for label in range(self.nrof_classes)])
            cur_inds = cur_inds.flatten()
            self.cursors = [min(c + 1, len(self.dataset)) for c in self.cursors]
            ind = np.random.choice(cur_inds, self.batch_size, p=self.probabilities)
            yield self.dataset[ind]

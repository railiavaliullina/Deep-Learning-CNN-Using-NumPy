from data.dataloaders.Sampler import Sampler
from data.dataloaders.Transform import *
from configs.config import cfg


class DataLoader(object):
    def __init__(self, dataset, nrof_classes, dataset_type, shuffle, batch_size,
                 sample_type, epoch_size=None, probabilities=None):
        """
        :param dataset (Dataset): объект класса Dataset.
        :param nrof_classes (int): количество классов в датасете.
        :param dataset_type (string): (['train', 'valid', 'test']).
        :param shuffle (bool): нужно ли перемешивать данные после очередной эпохи.
        :param batch_size (int): размер батча.
        :param sample_type (string): (['default' - берем последовательно все данные, 'balanced' - сбалансированно,
        'prob' - сэмплирем с учетом указанных вероятностей])
        :param epoch_size (int or None): размер эпохи. Если None, необходимо посчитать размер эпохи (=размеру обучающей выборки/batch_size)
        :param probabilities (array or None): в случае sample_type='prob' вероятности, с которыми будут выбраны элементы из каждого класса.
        """
        self.dataset = dataset
        self.nrof_classes = nrof_classes
        self.dataset_type = dataset_type
        self.shuffle = shuffle
        self.batch_size = batch_size
        self.sample_type = sample_type
        self.epoch_size = len(dataset) // batch_size if epoch_size is None else epoch_size
        self.probabilities = probabilities
        self.sampler = Sampler(dataset, nrof_classes, self.epoch_size, shuffle, batch_size, probabilities)

    def batch_generator(self):
        """
        Создание батчей на эпоху с учетом указанного размера эпохи и типа сэмплирования.
        """
        batch_generator = getattr(self.sampler, self.sample_type.name)()
        return batch_generator

    def plot_grid(self, images, labels, filename):
        """
        Создание сетки с изображениями в батче
        """
        n_rows, n_cols = int(self.batch_size / np.sqrt(self.batch_size)), int(
            self.batch_size / np.sqrt(self.batch_size))
        _, axs = plt.subplots(n_rows, n_cols, figsize=cfg["dataloader"]["show_batch"]["fig_size"])
        axs = axs.flatten()
        for i, (img, ax) in enumerate(zip(images, axs)):
            if len(img.shape) == 3:
                img = img.transpose((1, 2, 0))
            ax.imshow(img)
            ax.set_title(f'label: {labels[i]}')
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)
            ax.set_aspect('equal')
        if cfg["dataloader"]["show_batch"]["save_fig"]:
            plt.savefig(cfg["dataloader"]["show_batch"]["path_to_save"] + filename)
        if cfg["dataloader"]["show_batch"]["show_fig"]:
            plt.show()

    def show_batch(self):
        """
        Необходимо визуализировать и сохранить изображения в батче (один батч - одно окно). Предварительно привести значение в промежуток
        [0, 255) и типу к uint8
        """
        nrof_batches_to_save = cfg["dataloader"]["show_batch"]["nrof_batches_to_save"]
        dataset_ids = np.arange(len(self.dataset))
        random_ids = np.random.choice(dataset_ids, self.batch_size * nrof_batches_to_save, replace=False)

        for batch in range(nrof_batches_to_save):
            ids = random_ids[batch * self.batch_size: (batch + 1) * self.batch_size]
            images = self.dataset.images[ids]
            labels = self.dataset.labels[ids]

            images = [((im - np.min(im)) / (np.max(im) - np.min(im)) * 255).astype('uint8') for im in images]
            self.plot_grid(images, labels, filename=f'{cfg["dataset"]}_batch_{batch}.png')

import numpy as np
import matplotlib.pyplot as plt
import cv2
from skimage.transform import rotate
from skimage.util import random_noise

from configs.config import cfg
from data.dataloaders.Registry import REGISTRY_TYPE


class Transforms(object):
    def __init__(self, dataset_type, dataset_cfg):
        self.cfg = cfg
        self.dataset_type = dataset_type
        self.dataset_cfg = dataset_cfg
        self.saved_images = []
        self.saved_labels = []

    def show_image(self, image, title=''):
        image = (image - np.min(image)) / (np.max(image) - np.min(image))
        im_to_show = image
        if len(image.shape) > 2 and image.shape[-1] != 3:  # rgb
            im_to_show = image.transpose((1, 2, 0))
        plt.imshow(im_to_show)
        plt.title(title)
        plt.show()

    def clamp(self, image, to_255=True):
        image = (image - np.min(image)) / (np.max(image) - np.min(image))
        if to_255:
            return image * 255
        return image

    @REGISTRY_TYPE.register_module
    def Pad(self, image, image_size=None, fill=None, mode=None, show=False):
        """
        :param image_size (int or tuple): размер итогового изображения. Если одно число, на выходе будет
        квадратное изображение. Если 2 числа - прямоугольное.
        :param fill (int or tuple): значение, которым будет заполнены поля. Если одно число, все каналы будут заполнены
        этим числом. Если 3 - соответственно по каналам.
        :param mode (string): тип заполнения:
        constant: все поля будут заполнены значение fill;
        edge: все поля будут заполнены пикселями на границе;
        reflect: отображение изображения по краям (прим. [1, 2, 3, 4] => [3, 2, 1, 2, 3, 4, 3, 2])
        symmetric: симметричное отображение изображения по краям (прим. [1, 2, 3, 4] => [2, 1, 1, 2, 3, 4, 4, 3])
        """
        image = self.clamp(image)
        image_size = self.dataset_cfg["transform_parameters"][self.dataset_type]["Pad"][
            "image_size"] if image_size is None \
            else image_size
        mode = self.dataset_cfg["transform_parameters"][self.dataset_type]["Pad"]["mode"] if mode is None else mode
        fill = self.dataset_cfg["transform_parameters"][self.dataset_type]["Pad"]["fill"] if fill is None else fill

        padded_size = (image_size, image_size) if isinstance(image_size, int) else image_size
        padded_size_w, padded_size_h = padded_size[0], padded_size[1]
        image_size_w = image.shape[0] if len(image.shape) == 2 else image.shape[1]
        image_size_h = image.shape[1] if len(image.shape) == 2 else image.shape[2]
        assert padded_size_w >= image_size_w and padded_size_h >= image_size_h
        image_size_difference = ((padded_size_w - image_size_w)//2, (padded_size_h - image_size_h)//2)
        pad_width = image_size_difference

        if show:
            self.show_image(image, 'before Pad')
        self.saved_images.append(image)
        self.saved_labels.append('before all')

        if len(image.shape) > 2:
            channels = image.shape[0]
            if mode == 'constant':
                if fill is not None:
                    image = np.array(
                        [np.pad(image[c, :, :], pad_width=pad_width, mode=mode, constant_values=fill)
                         for c in range(channels)])
                else:
                    image = np.array(
                        [np.pad(image[c, :, :], pad_width=pad_width, mode=mode) for c in range(channels)])
            else:
                image = np.array([np.pad(image[c, :, :], pad_width=pad_width, mode=mode)
                                  for c in range(channels)])
        else:
            if fill is not None:
                image = np.pad(image, pad_width=pad_width, mode=mode, constant_values=fill)
            else:
                image = np.pad(image, pad_width=pad_width, mode=mode)

        image = self.clamp(image, to_255=False)
        if show:
            self.show_image(image, 'after Pad')
        self.saved_images.append(image)
        self.saved_labels.append('after Pad')
        return image

    @REGISTRY_TYPE.register_module
    def GaussianBlur(self, image, ksize=None, show=False):
        """
        :param ksize (tuple): размер фильтра.
        """
        image = self.clamp(image)
        if show:
            self.show_image(image, 'before GaussianBlur')

        ksize = self.dataset_cfg["transform_parameters"][self.dataset_type]["GaussianBlur"]["ksize"] if ksize is None \
            else ksize
        std = self.dataset_cfg["transform_parameters"][self.dataset_type]["GaussianBlur"]["std"]
        image = cv2.cvtColor(image.transpose((1, 2, 0)), cv2.COLOR_RGB2BGR) if len(image.shape) > 2 else image
        image = cv2.GaussianBlur(image, ksize, std)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).transpose((2, 0, 1)) if len(image.shape) > 2 else image
        image = self.clamp(image, to_255=False)
        if show:
            self.show_image(image, 'after GaussianBlur')
        self.saved_images.append(image)
        self.saved_labels.append('after GaussianBlur')
        return image

    @REGISTRY_TYPE.register_module
    def Normalize(self, image, mean=None, var=None, show=False):
        """
        :param mean (int or tuple): среднее значение (пикселя), которое необходимо вычесть.
        :param var (int): значение, на которое необходимо поделить.
        """
        image = self.clamp(image)
        if show:
            self.show_image(image, 'before Normalize')

        mean = self.dataset_cfg["transform_parameters"][self.dataset_type]["Normalize"][
            "mean"] if mean is None else mean
        var = self.dataset_cfg["transform_parameters"][self.dataset_type]["Normalize"]["var"] if var is None else var
        if len(image.shape) > 2:
            image = np.array([(image[c, :, :] - mean) / var for c in range(image.shape[0])])
        else:
            image = (image - mean) / var
        if show:
            self.show_image(image, 'after Normalize')
        self.saved_images.append(image)
        self.saved_labels.append('after Normalize')
        # image = self.clamp(image, to_255=False)
        return image

    @REGISTRY_TYPE.register_module
    def RandomRotateImage(self, image, min_angle=None, max_angle=None, show=False):
        """
        :param min_angle (int): минимальный угол поворота.
        :param max_angle (int): максимальный угол поворота.
        Угол поворота должен быть выбран равномерно из заданного промежутка.
        """
        if show:
            self.show_image(image, 'before RandomRotateImage')
        min_angle = self.dataset_cfg["transform_parameters"][self.dataset_type]["RandomRotateImage"]["min_angle"] \
            if min_angle is None else min_angle
        max_angle = self.dataset_cfg["transform_parameters"][self.dataset_type]["RandomRotateImage"]["max_angle"] \
            if max_angle is None else max_angle
        angle = np.random.choice(np.arange(min_angle, max_angle), 1)[0]
        if len(image.shape) > 2:
            image = np.array([rotate(image[c, :, :], angle) for c in range(image.shape[0])])
        else:
            image = rotate(image, angle)

        image = self.clamp(image, to_255=False)
        if show:
            self.show_image(image, 'after RandomRotateImage')
        self.saved_images.append(image)
        self.saved_labels.append('after RandomRotateImage')
        return image

    @REGISTRY_TYPE.register_module
    def GaussianNoise(self, image, mean=None, sigma=None, step=None, by_channel=False, show=False):
        """
        :param mean (int): среднее значение.
        :param sigma (int): максимальное значение ско. Итоговое значение должно быть выбрано равномерно в промежутке
        [0, sigma].
        :param by_channel (bool): если True, то по каналам независимо.
        """
        image = self.clamp(image, to_255=False)
        if show:
            self.show_image(image, 'before GaussianNoise')
        mean = self.dataset_cfg["transform_parameters"][self.dataset_type]["GaussianNoise"]["mean"] \
            if mean is None else mean
        sigma = self.dataset_cfg["transform_parameters"][self.dataset_type]["GaussianNoise"]["sigma"] \
            if sigma is None else sigma
        step = self.dataset_cfg["transform_parameters"][self.dataset_type]["GaussianNoise"]["step"] \
            if step is None else step
        sigma = np.random.choice(np.arange(0, sigma + step, step))
        by_channel = self.dataset_cfg["transform_parameters"][self.dataset_type]["GaussianNoise"]["by_channel"] \
            if by_channel is None else by_channel
        if by_channel:
            image = np.array([random_noise(image[c, :, :], mode='gaussian', mean=mean, var=sigma, clip=True)
                              for c in range(image.shape[0])])
        else:
            image = random_noise(image, mode='gaussian', mean=mean, var=sigma, clip=True)
        if show:
            self.show_image(image, 'after GaussianNoise')
        self.saved_images.append(image)
        self.saved_labels.append('after GaussianNoise')
        return image

    @REGISTRY_TYPE.register_module
    def Salt(self, image, prob=None, by_channel=False, show=False):
        """
        :param prob (float): вероятность, с которой пиксели будут заполнены белым.
        :param by_channel (bool): если True, то по каналам независимо.
        """
        image = self.clamp(image, to_255=False)
        if show:
            self.show_image(image, 'before Salt')
        prob = self.dataset_cfg["transform_parameters"][self.dataset_type]["Salt"]["prob"] \
            if prob is None else prob
        by_channel = self.dataset_cfg["transform_parameters"][self.dataset_type]["Salt"]["by_channel"] \
            if by_channel is None else by_channel
        if by_channel:
            image = np.array([random_noise(image[c, :, :], mode='salt', amount=prob, clip=True)
                              for c in range(image.shape[0])])
        else:
            image = random_noise(image, mode='salt', amount=prob, clip=True)
        if show:
            self.show_image(image, 'after Salt')
        self.saved_images.append(image)
        self.saved_labels.append('after Salt')
        return image

    @REGISTRY_TYPE.register_module
    def Pepper(self, image, prob=None, by_channel=False, show=False):
        """
        :param prob (float): вероятность, с которой пиксели будут заполнены черным.
        :param by_channel (bool): если True, то по каналам независимо.
        """
        image = self.clamp(image, to_255=False)
        if show:
            self.show_image(image, 'before Pepper')
        prob = self.dataset_cfg["transform_parameters"][self.dataset_type]["Pepper"]["prob"] \
            if prob is None else prob
        by_channel = self.dataset_cfg["transform_parameters"][self.dataset_type]["Pepper"]["by_channel"] \
            if by_channel is None else by_channel
        if by_channel:
            image = np.array([random_noise(image[c, :, :], mode='pepper', amount=prob, clip=True)
                              for c in range(image.shape[0])])
        else:
            image = random_noise(image, mode='pepper', amount=prob, clip=True)
        if show:
            self.show_image(image, 'after Pepper')
        self.saved_images.append(image)
        self.saved_labels.append('after Pepper')
        return image

    @REGISTRY_TYPE.register_module
    def Translate(self, image, shift=(10, 0), roll=True, show=False):
        """
        :param shift (int): количество пикселей, на которое необходимо сдвинуть изображение
        :param direction (string): направление (['right', 'left', 'down', 'up'])
        :param roll (bool): Если False, не заполняем оставшуюся часть. Если True, заполняем оставшимся краем.
        (прим. False: [1, 2, 3]=>[0, 1, 2]; True: [1, 2, 3] => [3, 1, 2])
        """
        if show:
            self.show_image(image, 'before Translate')
        shift = self.dataset_cfg["transform_parameters"][self.dataset_type]["Translate"][
            "shift"] if shift is None else shift
        roll = self.dataset_cfg["transform_parameters"][self.dataset_type]["Translate"][
            "roll"] if roll is None else roll
        border_mode = cv2.BORDER_CONSTANT if not roll else cv2.BORDER_WRAP
        transformation_matrix = np.float32([[1, 0, shift[0]], [0, 1, shift[1]]])
        if len(image.shape) > 2:
            image = np.array(
                [cv2.warpAffine(src=image[c, :, :], M=transformation_matrix, dsize=image[c, :, :].shape,
                                borderMode=border_mode, borderValue=0) for c in range(image.shape[0])])
        else:
            image = cv2.warpAffine(src=image, M=transformation_matrix, dsize=image.shape,
                                              borderMode=border_mode, borderValue=0)
        if show:
            self.show_image(image, 'after Translate')
        self.saved_images.append(image)
        self.saved_labels.append('after Translate')
        image = self.clamp(image, to_255=False)
        return image

    @REGISTRY_TYPE.register_module
    def CenterCrop(self, image, crop_size=None, show=False, save=True):
        """
        :param crop_size (int or tuple): размер вырезанного изображения (вырезать по центру).
        """
        if show:
            self.show_image(image, 'before CenterCrop')
        crop_size = self.dataset_cfg["transform_parameters"][self.dataset_type]["CenterCrop"]["crop_size"] \
            if crop_size is None else crop_size
        crop_size = (crop_size, crop_size) if isinstance(crop_size, int) else crop_size
        h, w = (image.shape[1], image.shape[2]) if len(image.shape) > 2 else (image.shape[0], image.shape[1])
        assert crop_size[0] <= h and crop_size[0] <= w, 'invalid crop size parameter'

        if len(image.shape) > 2:
            image = np.array([cv2.getRectSubPix(image[c, :, :], crop_size, (crop_size[0] / 2, crop_size[1] / 2)) for c in range(image.shape[0])])
        else:
            image = cv2.getRectSubPix(image, crop_size, (crop_size[0] / 2, crop_size[1] / 2))
        if show:
            self.show_image(image, 'after CenterCrop')
        if save:
            self.saved_images.append(image)
            self.saved_labels.append('after CenterCrop')
        image = self.clamp(image, to_255=False)
        return image

    @REGISTRY_TYPE.register_module
    def RandomCrop(self, image, crop_size=None, show=False):
        """
        :param crop_size (int or tuple): размер вырезанного изображения.
        """
        if show:
            self.show_image(image, 'before RandomCrop')
        crop_size = self.dataset_cfg["transform_parameters"][self.dataset_type]["RandomCrop"]["crop_size"] \
            if crop_size is None else crop_size
        crop_size = (crop_size, crop_size) if isinstance(crop_size, int) else crop_size
        h, w = (image.shape[1], image.shape[2]) if len(image.shape) > 2 else (image.shape[0], image.shape[1])
        assert crop_size[0] <= h and crop_size[0] <= w, 'invalid crop size parameter'

        # h_center, w_center = np.random.randint(crop_size[0]/2, h - crop_size[0]/2), \
        #        np.random.randint(crop_size[1]/2, w - crop_size[1]/2)
        if len(image.shape) > 2:
            image = np.array([cv2.getRectSubPix(image[c, :, :], crop_size, (crop_size[0] / 2, crop_size[1] / 2)) for c in range(image.shape[0])])
        else:
            image = cv2.getRectSubPix(image, crop_size, (crop_size[0] / 2, crop_size[1] / 2))
        if show:
            self.show_image(image, 'after RandomCrop')
        self.saved_images.append(image)
        self.saved_labels.append('after RandomCrop')
        image = self.clamp(image, to_255=False)
        return image

    @REGISTRY_TYPE.register_module
    def Scale(self, image, image_size=None, scale=None, show=False):
        """
        :param image_size (int): размер вырезанного изображения (по центру).
        :param scale (float): во сколько раз увеличить изображение.
        """
        if show:
            self.show_image(image, 'before Scale')
        image_size = self.dataset_cfg["transform_parameters"][self.dataset_type]["Scale"]["image_size"] \
            if image_size is None else image_size
        scale = self.dataset_cfg["transform_parameters"][self.dataset_type]["Scale"]["scale"] \
            if scale is None else scale
        center_cropped_image = self.CenterCrop(image, crop_size=image_size, show=False, save=False)

        h, w = (center_cropped_image.shape[1], center_cropped_image.shape[2]) if len(center_cropped_image.shape) > 2 \
            else (center_cropped_image.shape[0], center_cropped_image.shape[1])

        if len(image.shape) > 2:
            image = np.array([cv2.resize(center_cropped_image[c, :, :], (h * scale, w * scale)) for c in
                                     range(center_cropped_image.shape[0])])
        else:
            image = cv2.resize(center_cropped_image, (h * scale, w * scale))
        if show:
            self.show_image(image, 'after Scale')
        self.saved_images.append(image)
        self.saved_labels.append('after Scale')
        image = self.clamp(image, to_255=False)
        return image

    @REGISTRY_TYPE.register_module
    def ChangeBrightness(self, image, value=None, type=None, show=True):
        """
        :param value (int): насколько изменить яркость. Аналогично hue, contrast, saturation.
        :param type (string): один из [brightness, hue, contrast, saturation].
        """
        image = (image - np.min(image)) / (np.max(image) - np.min(image)) * 255
        if show:
            self.show_image(image, 'before ChangeBrightness')
        value = self.dataset_cfg["transform_parameters"][self.dataset_type]["ChangeBrightness"]["value"] \
            if value is None else value
        type = self.dataset_cfg["transform_parameters"][self.dataset_type]["ChangeBrightness"]["type"] \
            if type is None else type

        if type == 'brightness':
            image = cv2.convertScaleAbs(image, beta=value)

        elif type == 'contrast':
            image = cv2.convertScaleAbs(image, alpha=value)

        elif type in ['saturation', 'hue']:
            assert len(image.shape) == 3, 'invalid number of channels'
            image = cv2.cvtColor(image.transpose(1, 2, 0), cv2.COLOR_BGR2HSV)
            channel = 1 if type == 'saturation' else 0
            image[:, :, channel] = image[:, :, channel] + value
            image = cv2.cvtColor(image, cv2.COLOR_HSV2BGR).transpose((2, 0, 1))
        else:
            raise Exception
        image = self.clamp(image, to_255=False)
        if show:
            self.show_image(image, 'after ChangeBrightness')
        self.saved_images.append(image)
        self.saved_labels.append('after ChangeBrightness')

        return image

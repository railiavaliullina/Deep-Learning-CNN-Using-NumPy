
cfg = {
    "path": {
        "train": "E:/datasets/cifar-10-batches-py/",
        "test": "E:/datasets/cifar-10-batches-py/test_batch"
    },
    "classes": 10,
    "save_image_augmentations": False,  # saving image after every augmentation
    "save_batch_augmentations": True,  # saving batch after random augmentations (before training)
    "transform_parameters": {
            "show_at_every_transform": False,
            "train": {
                "Pad": {
                    "image_size": 33,  # (int or tuple)
                    "fill": 0,  # (int or tuple)
                    "mode": 'constant',  # ['constant', 'edge', 'reflect', 'symmetric']
                    "p": 0.5
                        },
                "GaussianBlur": {
                    "ksize": (3, 3),
                    "std": 1,
                    "p": 0.1
                },
                "RandomRotateImage": {
                    "min_angle": 5,
                    "max_angle": 45,
                    "p": 0.5
                },
                "GaussianNoise": {
                    "mean": 0,
                    "sigma": 0.004,
                    "step": 0.001,
                    "by_channel": True,
                    "p": 0.5
                },
                "Salt": {
                    "prob": 0.015,
                    "by_channel": True,
                    "p": 0.4
                },
                "Pepper": {
                    "prob": 0.015,
                    "by_channel": True,
                    "p": 0.4
                },
                "Translate": {
                    "shift": (1, 0),  # tuple (x_axis, y_axis)
                    "roll": True,
                    "p": 0.1
                },
                "CenterCrop": {
                    "crop_size": (26, 26),  # (26, 26),
                    "p": 0.2
                },
                "RandomCrop": {
                    "crop_size": (24, 24),  # (24, 24),
                    "p": 0.1
                },
                "Scale": {
                    "image_size": (20, 20),
                    "scale": 2,
                    "p": 0.1
                },
                "ChangeBrightness": {
                    "value": 2,
                    "type": 'brightness',  # ['saturation', 'contrast', 'brightness', 'hue']
                    "p": 0.1
                },
                "Normalize": {
                    "mean": 128,
                    "var": 128,  # 255
                    "p": 1.0
                }
            },
            "test": {
                "CenterCrop": {
                    "crop_size": (26, 26),
                    "p": 1.0
                },
                "Normalize": {
                    "mean": 128,
                    "var": 255,
                    "p": 1.0
                }
            }
    }
}

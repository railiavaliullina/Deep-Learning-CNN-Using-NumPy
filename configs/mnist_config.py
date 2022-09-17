cfg = {
    "path": {
        "train": "E:/datasets/MNIST/mnist.pkl.gz",
        "test": "E:/datasets/MNIST/mnist.pkl.gz"
    },
    "classes": 10,
    "save_image_augmentations": False,  # saving image after every augmentation
    "save_batch_augmentations": False,  # saving batch after random augmentations (before training)
    "transform_parameters": {
        "show_at_every_transform": False,
        "train": {
        },
        "train_with_aug":
            {
                "GaussianBlur": {
                    "ksize": (3, 3),
                    "std": 1.4,
                    "p": 0.2
                },
                "RandomRotateImage": {
                    "min_angle": 10,
                    "max_angle": 30,
                    "p": 0.5
                },
                "GaussianNoise": {
                    "mean": 0,
                    "sigma": 0.03,
                    "step": 0.01,
                    "by_channel": True,
                    "p": 0.3
                },
                "Salt": {
                    "prob": 0.015,
                    "by_channel": False,
                    "p": 0.4
                },
                "Pepper": {
                    "prob": 0.04,
                    "by_channel": False,
                    "p": 0.3
                },
                "Translate": {
                    "shift": (3, 0),  # tuple (x_axis, y_axis)
                    "roll": True,
                    "p": 0.4
                },
                "RandomCrop": {
                    "crop_size": (24, 24),
                    "p": 0.1,
                },
                "Pad": {
                    "image_size": 28,  # (int or tuple)
                    "fill": None,  # 0,  # (int or tuple)
                    "mode": 'symmetric',  # ['constant', 'edge', 'reflect', 'symmetric']
                    "p": 1.0
                },
            },
        "test": {
        }
    }
}

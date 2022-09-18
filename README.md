# Deep-Learning-CNN-Using-NumPy

## About The Project

1) CNN was implemented in Python + NumPy, without the use of frameworks for deep neural networks,
2) was trained on mnist, cifar-10 datasets.


## Implemented modules

• forward and backward pass for convolutional and pooling (max, average) layers;

• activation functions (ReLU, sigmoid, tanh, linear, ...),

• Softmax function,

• Optimizer class,

• objective function CrossEntropy,

• Dataset class,

• class DataLoader,

• batch_generator function, taking into account various sampling methods,

• data augmentation classes (pad, crop, scale, translate, rotate, noise, salt, pepper, hue, brightness, saturation, contrast, blur),

• unit tests,

• class "container" of the model,

• overfitting on one batch,

• all calculations for a batch, aggregation by average.


## Getting Started

Files to run:

    for training without augmentation -- homeworks/Homework_CNN_train_net.py
    for training with augmentation -- homeworks/Homework_CNN_train_net_with_aug.py


Графики точности на обучающей и тестовой выборках, лосса - в папке /plots.
Параметры обучения в конфиге configs/config.py

Время обучения одной эпохи:
1) без аугментации ~ 9.856 min
2) с аугментацией ~ 10.417 min

Время валидации:
1) без аугментации: ~ 1.134 min (на обучающей выборке), ~ 0.188 min (на тестовой выборке)
2) с аугментацией: ~ 1.777 min (на обучающей выборке), ~ 0.18 min min (на тестовой выборке)

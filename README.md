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


## Additional Information

Visualization of accuracy on the training and test samples, loss are in:

        /plots/
        
Training parameters are in:

    configs/config.py

Training time for one epoch:
1) without augmentation ~ 9.856 min
2) with augmentation ~ 10.417 min

Validation time:
1) without augmentation: ~ 1.134 min (on the training set), ~ 0.188 min (on the test set)
2) with augmentation: ~ 1.777 min (on the training set), ~ 0.18 min min (on the test set)

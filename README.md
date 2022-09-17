# Deep-Learning-CNN-Using-NumPy

Convolutional Neural Network:
1) Реализован на Python + NumPy, без использования фреймворков для глубоких нейронных сетей,
2) Обучен на датасетах mnist, cifar-10.


Реализовано следующее:

• forward и backward pass для convolutional и pooling (max, average) слоев;

• функции активации (ReLU, sigmoid, tanh, linear, ...),

• функция Softmax,

• класс Optimizer,

• целевая функция CrossEntropy,

• класс Dataset,

• класс DataLoader,

• функция batch_generator с учетом различных методов сэмплирования,

• классы аугментации данных (pad, crop, scale, translate, rotate, noise, salt, pepper, hue, brightness, saturation, contrast, blur),

• unit тесты,

• класс «контейнер» модели,

• оверфиттинг на одном батче,

• все вычисления для батча, агрегация средним.

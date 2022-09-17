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


Файлы для запуска: 
1) обучение без аугментации - homeworks/Homework_CNN_train_net.py
2) обучение с аугментацией - homeworks/Homework_CNN_train_net_with_aug.py

Графики точности на обучающей и тестовой выборках, лосса - в папке /plots.
Параметры обучения в конфиге configs/config.py

Время обучения одной эпохи:
1) без аугментации ~ 9.856 min
2) с аугментацией ~ 10.417 min

Время валидации:
1) без аугментации: ~ 1.134 min (на обучающей выборке), ~ 0.188 min (на тестовой выборке)
2) с аугментацией: ~ 1.777 min (на обучающей выборке), ~ 0.18 min min (на тестовой выборке)

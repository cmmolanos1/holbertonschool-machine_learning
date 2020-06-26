#!/usr/bin/env python3
"""
Transfer Learning.
"""
import tensorflow.keras as K
import numpy as np


def preprocess_data(X, Y):
    """Preprocess data for Resnet50.
    Args:
        X (np.ndarray): matrix of shape (m, 32, 32, 3) containing the CIFAR 10
                        data, where m is the number of data points.
        Y (np.ndarray): matrix of shape (m,) containing the CIFAR 10 labels
                        for X.
    Returns:
        X is a numpy.ndarray containing the preprocessed X
        Y is a numpy.ndarray containing the preprocessed Y
    """
    X = K.applications.resnet50.preprocess_input(X)
    Y = K.utils.to_categorical(Y, 10)

    return X, Y


if __name__ == '__main__':
    # Divide the data in Train and Test Datasets
    (x_train, y_train), (x_test, y_test) = K.datasets.cifar10.load_data()

    # Preprocess Data
    x_train, y_train = preprocess_data(x_train, y_train)
    x_test, y_test = preprocess_data(x_test, y_test)

    # Define the CNN base model
    model = K.applications.ResNet50(include_top=False,
                                    weights='imagenet',
                                    input_shape=(224, 224, 3))

    model_1 = K.Sequential()
    model_1.add(K.layers.UpSampling2D((7, 7)))
    model_1.add(model)
    model_1.add(K.layers.AveragePooling2D(pool_size=7))
    model_1.add(K.layers.Flatten())
    model_1.add(K.layers.Dense(10, activation=('softmax')))

    checkpoint = K.callbacks.ModelCheckpoint(filepath='cifar10.h5',
                                             monitor='val_acc',
                                             mode='max',
                                             verbose=1,
                                             save_best_only=True)

    model_1.compile(optimizer=K.optimizers.RMSprop(learning_rate=1e-4),
                    loss='categorical_crossentropy',
                    metrics=['acc'])

    model_1.fit(x_train, y_train,
                validation_data=(x_test, y_test),
                batch_size=32,
                epochs=5,
                verbose=1,
                callbacks=[checkpoint])

    model_1.save('cifar10.h5')

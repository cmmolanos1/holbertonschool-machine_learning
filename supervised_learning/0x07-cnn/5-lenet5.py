#!/usr/bin/env python3
"""
Pooling backward
"""

import tensorflow.keras as K


def lenet5(X):
    """Builds a modified version of the LeNet-5 architecture using keras.

    Args:
        X (K.Input): shape (m, 28, 28, 1) containing the input images for the
                     network

    Returns:
        A K.Model compiled to use Adam optimization (with default
        hyperparameters) and accuracy metrics.
    """
    C1 = K.layers.Conv2D(filters=6,
                         kernel_size=5,
                         padding='same',
                         activation='relu',
                         kernel_initializer='he_normal')(X)

    S2 = K.layers.MaxPooling2D(pool_size=(2, 2),
                               strides=(2, 2))(C1)

    C3 = K.layers.Conv2D(filters=16,
                         kernel_size=5,
                         padding='valid',
                         activation='relu',
                         kernel_initializer='he_normal')(S2)

    S4 = K.layers.MaxPooling2D(pool_size=(2, 2),
                               strides=(2, 2))(C3)

    flatten = K.layers.Flatten()(S4)

    C5 = K.layers.Dense(units=120,
                        activation='relu',
                        kernel_initializer='he_normal')(flatten)

    F6 = K.layers.Dense(units=84,
                        activation='relu',
                        kernel_initializer='he_normal')(C5)

    OUTPUT = K.layers.Dense(units=10,
                            activation='softmax',
                            kernel_initializer='he_normal')(F6)

    model = K.Model(inputs=X, outputs=OUTPUT)

    model.compile(optimizer=K.optimizers.Adam(),
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    return model

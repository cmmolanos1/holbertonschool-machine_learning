#!/usr/bin/env python3
"""
Resnet 50
"""

import tensorflow.keras as K

dense_block = __import__('5-dense_block').dense_block
transition_layer = __import__('6-transition_layer').transition_layer


def densenet121(growth_rate=32, compression=1.0):
    """Builds the DenseNet-121 architecture as described in Densely Connected
    Convolutional Networks.

    Args:
        growth_rate (int): the growth rate.
        compression (float): the compression factor.

    Returns:
        The keras model.
    """
    X_input = K.layers.Input(shape=(224, 224, 3))

    X = K.layers.BatchNormalization()(X_input)
    X = K.layers.Activation('relu')(X)
    X = K.layers.Conv2D(filters=2*growth_rate,
                        kernel_size=7,
                        padding='same',
                        strides=2,
                        kernel_initializer='he_normal')(X)

    X = K.layers.MaxPooling2D(pool_size=(3, 3),
                              padding='same',
                              strides=(2, 2))(X)

    X, nb_filters = dense_block(X, 2 * growth_rate, growth_rate, 6)
    X, nb_filters = transition_layer(X, nb_filters, compression)
    X, nb_filters = dense_block(X, nb_filters, growth_rate, 12)
    X, nb_filters = transition_layer(X, nb_filters, compression)
    X, nb_filters = dense_block(X, nb_filters, growth_rate, 24)
    X, nb_filters = transition_layer(X, nb_filters, compression)
    X, nb_filters = dense_block(X, nb_filters, growth_rate, 16)

    X = K.layers.AvgPool2D(pool_size=(7, 7),
                           padding='same')(X)

    X = K.layers.Dense(units=1000,
                       activation='softmax',
                       kernel_initializer='he_normal')(X)

    model = K.Model(inputs=X_input, outputs=X)

    return model

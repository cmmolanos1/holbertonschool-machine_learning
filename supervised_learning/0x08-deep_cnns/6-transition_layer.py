#!/usr/bin/env python3
"""
Dense Block
"""

import tensorflow.keras as K


def transition_layer(X, nb_filters, compression):
    """Builds a transition layer as described in Densely Connected
    Convolutional Networks.

    Args:
        X (Keras layer): output from the previous layer.
        nb_filters (int): number of filters in X.
        compression (float): compression factor for the transition layer.

    Returns:
         The output of the transition layer and the number of filters within
         the output, respectively.
    """
    nb_filters = int(nb_filters * compression)

    x = K.layers.BatchNormalization()(X)
    x = K.layers.Activation('relu')(x)
    x = K.layers.Conv2D(filters=nb_filters,
                        kernel_size=1,
                        padding='same',
                        kernel_initializer='he_normal')(x)

    x = K.layers.AvgPool2D(pool_size=(2, 2),
                           strides=(2, 2),
                           padding='valid')(x)

    return x, nb_filters

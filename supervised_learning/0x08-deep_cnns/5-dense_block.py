#!/usr/bin/env python3
"""
Dense Block
"""

import tensorflow.keras as K


def bottlenecks(inputs, growth_rate):
    """Calculus for bottlenecks layers.

    Args:
        inputs (Keras layer): output from the previous layer.
        growth_rate (int): growth rate for the dense block.

    Returns:
        output of the bottlenecks
    """
    x = K.layers.BatchNormalization()(inputs)
    x = K.layers.Activation('relu')(x)
    x = K.layers.Conv2D(filters=4 * growth_rate,
                        kernel_size=1,
                        padding='same',
                        kernel_initializer='he_normal')(x)

    x = K.layers.BatchNormalization()(x)
    x = K.layers.Activation('relu')(x)
    x = K.layers.Conv2D(filters=growth_rate,
                        kernel_size=(3, 3),
                        padding='same',
                        kernel_initializer='he_normal')(x)
    return x


def dense_block(X, nb_filters, growth_rate, layers):
    """Builds a dense block as described in Densely Connected Convolutional
    Networks.

    Args:
        X (Keras layer): output from the previous layer.
        nb_filters (int): an integer representing the number of filters in X.
        growth_rate (int): growth rate for the dense block.
        layers (int):  number of layers in the dense block

    Returns:
        The concatenated output of each layer within the Dense Block and the
        number of filters within the concatenated outputs, respectively.
    """
    for i in range(layers):
        conv_outputs = bottlenecks(X, growth_rate)
        X = K.layers.concatenate([X, conv_outputs])
        nb_filters += growth_rate

    return X, nb_filters

#!/usr/bin/env python3
"""
Projection block.
"""

import tensorflow.keras as K


def projection_block(A_prev, filters, s=2):
    """Builds a projection block as described in Deep Residual Learning for
    Image Recognition (2015).

    Args:
        A_prev (Keras layer): the output from the previous layer.
        filters (list or tuple): [F11, F3, F12]
                                 F11 is the number of filters in the first 1x1
                                 convolution.
                                 F3 is the number of filters in the 3x3
                                 convolution.
                                 F12 is the number of filters in the second
                                 1x1 convolution as well as the 1x1
                                 convolution in the shortcut connection.
        s (int): stride of the first convolution in both the main path and the
                 shortcut connection.
    Returns:
        The activated output of the identity block.
    """
    X = K.layers.Conv2D(filters=filters[0],
                        kernel_size=1,
                        padding='same',
                        strides=(s, s),
                        kernel_initializer='he_normal')(A_prev)

    X = K.layers.BatchNormalization(axis=3)(X)

    X = K.layers.Activation('relu')(X)

    X = K.layers.Conv2D(filters=filters[1],
                        kernel_size=3,
                        padding='same',
                        # strides=(s, s),
                        kernel_initializer='he_normal')(X)

    X = K.layers.BatchNormalization()(X)

    X = K.layers.Activation('relu')(X)

    X = K.layers.Conv2D(filters=filters[2],
                        kernel_size=1,
                        padding='same',
                        # strides=(s, s),
                        kernel_initializer='he_normal')(X)

    X = K.layers.BatchNormalization()(X)

    # Shortcut connection.
    shortcut = K.layers.Conv2D(filters=filters[2],
                               kernel_size=1,
                               padding='same',
                               strides=(s, s),
                               kernel_initializer='he_normal')(A_prev)

    shortcut = K.layers.BatchNormalization()(shortcut)

    adding = K.layers.Add()([X, shortcut])

    output = K.layers.Activation('relu')(adding)

    return output

#!/usr/bin/env python3

import tensorflow.keras as K


def identity_block(A_prev, filters):
    """Builds an identity block as described in Deep Residual Learning for
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
    Returns:
        The activated output of the identity block.
    """
    conv1 = K.layers.Conv2D(filters=filters[0],
                            kernel_size=1,
                            padding='same',
                            kernel_initializer='he_normal')(A_prev)

    batch1 = K.layers.BatchNormalization()(conv1)

    relu1 = K.layers.Activation('relu')(batch1)

    conv2 = K.layers.Conv2D(filters=filters[1],
                            kernel_size=3,
                            padding='same',
                            kernel_initializer='he_normal')(relu1)

    batch2 = K.layers.BatchNormalization()(conv2)

    relu2 = K.layers.Activation('relu')(batch2)

    conv3 = K.layers.Conv2D(filters=filters[2],
                            kernel_size=1,
                            padding='same',
                            kernel_initializer='he_normal')(relu2)

    batch3 = K.layers.BatchNormalization()(conv3)

    adding = K.layers.Add()([batch3, A_prev])

    output = K.layers.Activation('relu')(adding)

    return output

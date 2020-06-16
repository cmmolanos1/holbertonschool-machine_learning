#!/usr/bin/env python3
"""
Inception Network
"""

import tensorflow.keras as K

inception_block = __import__('0-inception_block').inception_block


def inception_network():
    """Builds the inception network as described in Going Deeper with
    Convolutions (2014):

    Returns:
         the keras model
    """
    X = K.layers.Input(shape=(224, 224, 3))

    conv1 = K.layers.Conv2D(filters=64,
                            kernel_size=7,
                            strides=2,
                            padding='same',
                            kernel_initializer='he_normal',
                            activation='relu')(X)

    maxpool1 = K.layers.MaxPooling2D(pool_size=(3, 3),
                                     padding='same',
                                     strides=(2, 2))(conv1)

    conv2_1 = K.layers.Conv2D(filters=64,
                              kernel_size=1,
                              strides=1,
                              padding='same',
                              kernel_initializer='he_normal',
                              activation='relu')(maxpool1)

    conv2_2 = K.layers.Conv2D(filters=192,
                              kernel_size=3,
                              strides=1,
                              padding='same',
                              kernel_initializer='he_normal',
                              activation='relu')(conv2_1)

    maxpool2 = K.layers.MaxPooling2D(pool_size=(3, 3),
                                     padding='same',
                                     strides=(2, 2))(conv2_2)

    inception3a = inception_block(maxpool2, (64, 96, 128, 16, 32, 32))

    inception3b = inception_block(inception3a, (128, 128, 192, 32, 96, 64))

    maxpool3 = K.layers.MaxPooling2D(pool_size=(3, 3),
                                     padding='same',
                                     strides=(2, 2))(inception3b)

    inception4a = inception_block(maxpool3, (192, 96, 208, 16, 48, 64))

    inception4b = inception_block(inception4a, (160, 112, 224, 24, 64, 64))

    inception4c = inception_block(inception4b, (128, 128, 256, 24, 64, 64))

    inception4d = inception_block(inception4c, (112, 144, 288, 32, 64, 64))

    inception4e = inception_block(inception4d, (256, 160, 320, 32, 128, 128))

    maxpool4 = K.layers.MaxPooling2D(pool_size=(3, 3),
                                     padding='same',
                                     strides=(2, 2))(inception4e)

    inception5a = inception_block(maxpool4, (256, 160, 320, 32, 128, 128))

    inception5b = inception_block(inception5a, (384, 192, 384, 48, 128, 128))

    avgpool1 = K.layers.AvgPool2D(pool_size=(7, 7),
                                  strides=(1, 1),
                                  padding='valid')(inception5b)

    dropout = K.layers.Dropout(0.4)(avgpool1)

    # linear = K.layers.Dense(units=1000,
    #                         activation='relu',
    #                         kernel_initializer='he_normal')(dropout)

    softmax = K.layers.Dense(units=1000,
                             activation='softmax',
                             kernel_initializer='he_normal')(dropout)

    model = K.Model(inputs=X, outputs=softmax)

    return model

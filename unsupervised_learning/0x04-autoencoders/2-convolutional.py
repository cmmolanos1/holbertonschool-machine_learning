#!/usr/bin/env python3
"""
Vanilla Encoder
"""

import tensorflow.keras as K


def autoencoder(input_dims, filters, latent_dims):
    """creates a convolutional autoencoder:

    input_dims is a tuple of integers containing the dimensions of the model
    input
    filters is a list containing the number of filters for each convolutional
    layer in the encoder, respectively
        the filters should be reversed for the decoder
    latent_dims is a tuple of integers containing the dimensions of the latent
    space representation
    Each convolution in the encoder should use a kernel size of (3, 3) with
    same padding and relu activation, followed by max pooling of size (2, 2)
    Each convolution in the decoder, except for the last two, should use a
    filter size of (3, 3) with same padding and relu activation, followed by
    upsampling of size (2, 2)
        The second to last convolution should instead use valid padding
        The last convolution should have the same number of filters as the
        number of channels in input_dims with sigmoid activation and no
        upsampling
    Returns: encoder, decoder, auto
        encoder is the encoder model
        decoder is the decoder model
        auto is the full autoencoder model
    """
    input_encoder = K.Input(shape=input_dims)

    encoded = K.layers.Conv2D(filters=filters[0],
                              kernel_size=3,
                              padding='same',
                              activation='relu')(input_encoder)
    encoded_pool = K.layers.MaxPool2D(
        pool_size=(2, 2), padding='same')(encoded)

    for i in range(1, len(filters)):
        encoded = K.layers.Conv2D(filters=filters[i],
                                  kernel_size=3,
                                  padding='same',
                                  activation='relu')(encoded_pool)
        encoded_pool = K.layers.MaxPool2D(
            pool_size=(2, 2), padding='same')(encoded)

    latent = encoded_pool

    encoder = K.models.Model(input_encoder, latent)
    encoder.summary()

    input_decoder = K.Input(shape=(latent_dims))
    decoded = K.layers.Conv2D(filters=filters[i], kernel_size=3,
                              padding='same',
                              activation='relu')(input_decoder)

    decoded_pool = K.layers.UpSampling2D(size=[2, 2])(decoded)
    for i in range(len(filters) - 2, -1, -1):
        if i == 0:
            decoded = K.layers.Conv2D(filters=filters[i], kernel_size=3,
                                      padding='valid',
                                      activation='relu')(decoded_pool)
            decoded_pool = K.layers.UpSampling2D(size=[2, 2])(decoded)
        else:
            decoded = K.layers.Conv2D(filters=filters[i], kernel_size=3,
                                      padding='same',
                                      activation='relu')(decoded_pool)
            decoded_pool = K.layers.UpSampling2D(size=[2, 2])(decoded)
    decoded = K.layers.Conv2D(filters=1, kernel_size=3,
                              padding='same',
                              activation='sigmoid')(decoded_pool)

    decoder = K.models.Model(input_decoder, decoded)
    decoder.summary()

    input_autoencoder = K.Input(shape=(input_dims))
    encoder_outs = encoder(input_autoencoder)
    decoder_outs = decoder(encoder_outs)

    autoencoder = K.models.Model(
        inputs=input_autoencoder, outputs=decoder_outs)

    autoencoder.compile(optimizer='Adam', loss='binary_crossentropy')

    return encoder, decoder, autoencoder

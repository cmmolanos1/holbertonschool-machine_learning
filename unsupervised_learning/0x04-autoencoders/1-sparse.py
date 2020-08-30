#!/usr/bin/env python3
"""
Vanilla Encoder
"""

import tensorflow.keras as keras


def sparse(input_dims, hidden_layers, latent_dims, lambtha):
    """creates a sparse autoencoder:

    input_dims is an integer containing the dimensions of the model input
    hidden_layers is a list containing the number of nodes for each hidden
    layer in the encoder, respectivelythe hidden layers should be reversed
    for the decoder
    latent_dims is an integer containing the dimensions of the latent space
    representation
    lambtha is the regularization parameter used for L1 regularization on
    the encoded output
    Returns: encoder, decoder, auto
        encoder is the encoder model
        decoder is the decoder model
        auto is the sparse autoencoder model

    """
    input_img = keras.Input(shape=(input_dims,))
    for i, layer in enumerate(hidden_layers):
        if i == 0:
            encoded = keras.layers.Dense(layer, activation='relu')(input_img)
        else:
            encoded = keras.layers.Dense(layer, activation='relu')(encoded)

    botneckle = keras.layers.Dense(
        latent_dims, activation='relu',
        activity_regularizer=keras.regularizers.l1(lambtha))(encoded)

    encoder = keras.models.Model(input_img, botneckle)

    input_botneckle = keras.Input(shape=(latent_dims,))
    for i in range(len(hidden_layers) - 1, -1, -1):
        if i == len(hidden_layers) - 1:
            decoded = keras.layers.Dense(
                hidden_layers[i], activation='relu')(input_botneckle)
        else:
            decoded = keras.layers.Dense(
                hidden_layers[i], activation='relu')(decoded)
    decoded = keras.layers.Dense(input_dims, activation='sigmoid')(decoded)

    decoder = keras.models.Model(input_botneckle, decoded)

    input_autoencoder = keras.Input(shape=(input_dims,))
    encoder_outs = encoder(input_autoencoder)
    decoder_outs = decoder(encoder_outs)

    autoencoder = keras.models.Model(
        inputs=input_autoencoder, outputs=decoder_outs)

    autoencoder.compile(optimizer='Adam', loss='binary_crossentropy')

    return encoder, decoder, autoencoder

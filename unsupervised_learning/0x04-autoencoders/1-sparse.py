#!/usr/bin/env python3
"""
Vanilla Encoder
"""

import tensorflow.keras as K


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
    input_img = K.Input(shape=(input_dims,))
    for i, layer in enumerate(hidden_layers):
        if i == 0:
            encoded = K.layers.Dense(layer, activation='relu')(input_img)
        else:
            encoded = K.layers.Dense(layer, activation='relu')(encoded)

    botneckle = K.layers.Dense(
        latent_dims, activation='relu',
        activity_regularizer=K.regularizers.l1(lambtha))(encoded)

    encoder = K.models.Model(input_img, botneckle)

    input_botneckle = K.Input(shape=(latent_dims,))
    for i in range(len(hidden_layers) - 1, -1, -1):
        if i == len(hidden_layers) - 1:
            decoded = K.layers.Dense(
                hidden_layers[i], activation='relu')(input_botneckle)
        else:
            decoded = K.layers.Dense(
                hidden_layers[i], activation='relu')(decoded)
    decoded = K.layers.Dense(input_dims, activation='sigmoid')(decoded)

    decoder = K.models.Model(input_botneckle, decoded)

    input_autoencoder = K.Input(shape=(input_dims,))
    encoder_outs = encoder(input_autoencoder)
    decoder_outs = decoder(encoder_outs)

    autoencoder = K.models.Model(
        inputs=input_autoencoder, outputs=decoder_outs)

    autoencoder.compile(optimizer='Adam', loss='binary_crossentropy')

    return encoder, decoder, autoencoder

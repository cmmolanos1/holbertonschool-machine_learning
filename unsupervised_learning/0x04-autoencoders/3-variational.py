#!/usr/bin/env python3
"""
Vanilla Encoder
"""

import tensorflow.keras as keras


def autoencoder(input_dims, hidden_layers, latent_dims):
    """creates a variational autoencoder:

    input_dims is an integer containing the dimensions of the model input
    hidden_layers is a list containing the number of nodes for each hidden
    layer in the encoder, respectively
        the hidden layers should be reversed for the decoder
    latent_dims is an integer containing the dimensions of the latent space
    representation
    Returns: encoder, decoder, auto
        encoder is the encoder model, which should output the latent
        representation, the mean, and the log variance, respectively
        decoder is the decoder model
        auto is the full autoencoder model
    """
    input_x = keras.Input(shape=(input_dims,))
    for i, layer in enumerate(hidden_layers):
        if i == 0:
            encoded = keras.layers.Dense(layer, activation='relu')(input_x)
        else:
            encoded = keras.layers.Dense(layer, activation='relu')(encoded)

    z_mean = keras.layers.Dense(latent_dims)(encoded)
    z_stand_des = keras.layers.Dense(latent_dims)(encoded)

    def sampling(args):
        z_mean, z_stand_des = args
        epsilon = keras.backend.random_normal(shape=(latent_dims,),
                                              mean=0.0, stddev=1.0)
        return z_mean + keras.backend.exp(z_stand_des) * epsilon

    z = keras.layers.Lambda(sampling, output_shape=(
        latent_dims,))([z_mean, z_stand_des])

    encoder = keras.models.Model(input_x, z)
    encoder.summary()

    input_z = keras.Input(shape=(latent_dims,))
    for i in range(len(hidden_layers) - 1, -1, -1):
        if i == len(hidden_layers) - 1:
            decoded = keras.layers.Dense(
                hidden_layers[i], activation='relu')(input_z)
        else:
            decoded = keras.layers.Dense(
                hidden_layers[i], activation='relu')(decoded)
    decoded = keras.layers.Dense(input_dims, activation='sigmoid')(decoded)
    decoder = keras.models.Model(input_z, decoded)
    decoder.summary()
    x = keras.Input(shape=(input_dims,))
    z_encoder = encoder(x)
    x_decoder_mean = decoder(z_encoder)

    autoencoder = keras.models.Model(
        inputs=x, outputs=x_decoder_mean)
    autoencoder.summary()

    def vae_loss(x, x_decoder_mean):
        x_loss = keras.backend.binary_crossentropy(x, x_decoder_mean)
        kl_loss = - 0.5 * keras.backend.mean(1 + z_stand_des -
                                             keras.backend.square(z_mean) -
                                             keras.backend.exp(z_stand_des),
                                             axis=-1)
        return x_loss + kl_loss

    autoencoder.compile(optimizer='Adam', loss=vae_loss)

    return encoder, decoder, autoencoder

#!/usr/bin/env python3
"""
Sequential
"""
from tensorflow import keras
from tensorflow.python.keras.layers import Dense, Dropout, Input


def build_model(nx, layers, activations, lambtha, keep_prob):
    """Builds a neural network with the Keras library:

    Args:
        nx (int): number of input features to the network:
        layers (list): list containing the number of nodes in each layer of
                       the network.
        activations (list): list containing the activation functions used
                            for each layer of the network
        lambtha (float): L2 regularization parameter
        keep_prob (float): probability that a node will be kept for dropout

    Returns:
        the keras model
    """
    X_input = Input(shape=(nx,))

    X = Dense(layers[0],
              activation=activations[0],
              kernel_regularizer=keras.regularizers.l2(lambtha))(X_input)

    for nodes, act in zip(layers[1::], activations[1::]):
        X = Dropout(1 - keep_prob)(X)
        X = Dense(nodes,
                  activation=act,
                  kernel_regularizer=keras.regularizers.l2(lambtha))(X)

    model = keras.Model(inputs=X_input, outputs=X)
    return model

#!/usr/bin/env python3
"""
Sequential
"""
import tensorflow.keras as K


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
    model = K.models.Sequential()
    model.add(K.layers.Dense(layers[0],
                             input_dim=nx,
                             activation=activations[0],
                             kernel_regularizer=K.regularizers.l2(lambtha)))
    for nodes, act in zip(layers[1::], activations[1::]):
        model.add(K.layers.Dropout(1 - keep_prob))
        model.add(K.layers.
                  Dense(nodes,
                        activation=act,
                        kernel_regularizer=K.regularizers.l2(lambtha)))

    return model

#!/usr/bin/env python3
"""Forward propagation"""
import tensorflow as tf

create_layer = __import__('1-create_layer').create_layer


def forward_prop(x, layer_sizes=[], activations=[]):
    """Creates the forward propagation graph for the neural network.

    Args:
        x (tensor): placeholder for the input data.
        layer_sizes (list): list containing the number of nodes in each layer
                            of the network.
        activations (list): list containing the activation functions for each
                            layer of the network.

    Returns:
        tensor: prediction of neural network.
    """
    A = create_layer(x, layer_sizes[0], activations[0])
    for layer in range(1, len(layer_sizes)):
        A = create_layer(A, layer_sizes[layer], activations[layer])
    return A

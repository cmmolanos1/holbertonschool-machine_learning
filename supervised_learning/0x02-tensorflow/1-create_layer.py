#!/usr/bin/env python3
"""Create PlaceHolder"""
import tensorflow as tf


def create_layer(prev, n, activation):
    """ Create a NN layer.

    Args:
        prev (tensor): tensor output of the previous layer.
        n (int): number of nodes in the layer to create.
        activation (tf.nn.activation): activation function.

    Returns:
        tensor: the layer created with shape [?, n].

    """
    init = tf.contrib.layers.variance_scaling_initializer(mode="FAN_AVG")
    layer = tf.layers.Dense(units=n,
                            activation=activation,
                            # Weights
                            kernel_initializer=init,
                            name="layer")

    return layer(prev)

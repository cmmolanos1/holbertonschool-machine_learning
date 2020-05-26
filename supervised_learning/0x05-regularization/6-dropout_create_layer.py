#!/usr/bin/env python3
"""
DropOut Regularization
"""

import tensorflow as tf


def dropout_create_layer(prev, n, activation, keep_prob):
    """Creates a tensorflow layer that includes dropout regularization.

    Args:
        prev (tensor): contains the output of the previous layer.
        n (n):  number of nodes the new layer should contain.
        activation (tensor): the activation function that should be used
                             on the layer.
        keep_prob (float): probability that a node will be kept.

    Returns:
        tensor: the output of the new layer.
    """
    dropout = tf.layers.Dropout(keep_prob)
    init = tf.contrib.layers.variance_scaling_initializer(mode="FAN_AVG")
    layer = tf.layers.Dense(units=n,
                            activation=activation,
                            kernel_initializer=init,
                            kernel_regularizer=dropout)
    return layer(prev)

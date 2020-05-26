#!/usr/bin/env python3
"""
L2 Regularization
"""

import tensorflow as tf


def l2_reg_create_layer(prev, n, activation, lambtha):
    """Creates a tensorflow layer that includes L2 regularization.

    Args:
        prev (tensor): contains the output of the previous layer.
        n (n):  number of nodes the new layer should contain.
        activation (tensor): the activation function that should be used
                             on the layer.
        lambtha (float): the L2 regularization parameter.

    Returns:
        tensor: the output of the new layer.
    """
    reg = tf.contrib.layers.l2_regularizer(lambtha)
    init = tf.contrib.layers.variance_scaling_initializer(mode="FAN_AVG")
    layer = tf.layers.Dense(units=n,
                            activation=activation,
                            kernel_initializer=init,
                            kernel_regularizer=reg)
    return layer(prev)

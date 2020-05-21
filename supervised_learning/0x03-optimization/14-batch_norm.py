#!/usr/bin/env python3
"""
Batch normalization
"""

import tensorflow as tf


def create_batch_norm_layer(prev, n, activation):
    """Creates a batch normalization layer for a neural network.

    Args:
        prev (tensor): the activated output of the previous layer.
        n (int): number of nodes in the layer to be created.
        activation (tensor): activation function that should be used on the
                             output of the layer

    Returns:
         tensor of the activated output for the layer.

    """
    # X, W, b ---> Z
    init = tf.contrib.layers.variance_scaling_initializer(mode="FAN_AVG")
    hidden = tf.layers.Dense(units=n, kernel_initializer=init)
    Z = hidden(prev)

    # Z, Gamma, Beta ---> Z_tilde
    epsilon = 1e-8
    beta = tf.Variable(tf.constant(0.0, shape=[n]), name='beta',
                       trainable=True)
    gamma = tf.Variable(tf.constant(1.0, shape=[n]), name='gamma',
                        trainable=True)
    mean, var = tf.nn.moments(Z, axes=[0])
    Z_tilde = tf.nn.batch_normalization(Z, mean, var, beta, gamma, epsilon)

    # A = g(Z_tilde)
    if activation is None:
        return Z_tilde
    return activation(Z_tilde)

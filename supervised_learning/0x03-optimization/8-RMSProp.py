#!/usr/bin/env python3
"""
RMS
"""

import tensorflow as tf


def create_RMSProp_op(loss, alpha, beta2, epsilon):
    """Creates the training operation for a neural network in tensorflow
    using the RMSProp optimization algorithm.

    Args:
        loss (float): loss of the network.
        alpha (float): learning rate.
        beta2 (float): RMSProp weight.
        epsilon (float): a small number to avoid division by zero.

    Returns:
        tensor:  the RMSProp optimization operation.
    """
    optimizer = tf.train.RMSPropOptimizer(learning_rate=alpha,
                                          decay=beta2,
                                          epsilon=epsilon)
    train = optimizer.minimize(loss)
    return train

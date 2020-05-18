#!/usr/bin/env python3
"""
RMS
"""

import tensorflow as tf


def create_Adam_op(loss, alpha, beta1, beta2, epsilon):
    """Updates a variable in place using the Adam optimization algorithm.

    Args:
        loss (float):  loss of the network.
        alpha (float): learning rate.
        beta1 (float): weight used for the first moment.
        beta2 (float): weight used for the second moment.
        epsilon (float): small number to avoid division by zero.

    Returns:
        tensor: the Adam optimization operation
    """
    optimizer = tf.train.AdamOptimizer(alpha, beta1, beta2, epsilon)
    return optimizer.minimize(loss)

#!/usr/bin/env python3
"""
L2 Regularization
"""

import tensorflow as tf


def l2_reg_cost(cost):
    """Calculates the cost of a neural network with L2 regularization.

    Args:
        cost (tensor): contains the cost of the network without L2
                       regularization.

    Returns:
        tensor: the cost of the network accounting for L2 regularization.
    """
    return tf.contrib.layers.l2_regularizer(cost)

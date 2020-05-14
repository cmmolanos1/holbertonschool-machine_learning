#!/usr/bin/env python3
"""Train Op"""
import tensorflow as tf


def create_train_op(loss, alpha):
    """Creates the training operation for the network:

    Args:
        loss (tensor): the loss of the network’s prediction.
        alpha: learning rate.

    Returns:
        an operation that trains the network using gradient descent.
    """
    optimizer = tf.train.GradientDescentOptimizer(alpha)
    return optimizer.minimize(loss)

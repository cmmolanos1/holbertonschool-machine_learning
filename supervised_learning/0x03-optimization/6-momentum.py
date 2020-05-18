#!/usr/bin/env python3
"""
Momentum
"""

import tensorflow as tf


def create_momentum_op(loss, alpha, beta1):
    """Trains NN by using the gradient descent with momentum optimization
    algorithm.

    Args:
        loss (float): loss of the network.
        alpha (float): learning rate.
        beta1 (float): momentum weight.

    Returns:
        tensor: the momentum optimization operation.
    """
    optimizer = tf.train.MomentumOptimizer(alpha, beta1)
    return optimizer.minimize(loss)

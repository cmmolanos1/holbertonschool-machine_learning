#!/usr/bin/env python3
"""
Loss
"""

import tensorflow as tf


def calculate_loss(y, y_pred):
    """Calculates the cross-entropy loss of a prediction.

    Args:
        y (tensor): placeholder for the labels of the input data.
        y_pred (tensor): the network’s predictions.

    Returns:
        tensor: the loss of the prediction
    """
    return tf.losses.softmax_cross_entropy(y, y_pred)

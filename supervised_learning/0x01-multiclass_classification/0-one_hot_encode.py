#!/usr/bin/env python3

import numpy as np


def one_hot_encode(Y, classes):
    """that converts a numeric label vector into a one-hot matrix

    Args:
        Y (numpy.ndarray): numeric class labels.
        classes (int): maximum number of classes.

    Returns:
        numpy.ndarray: one hot matrix.
    """
    if Y[Y < 0] or Y[Y >= classes] or len(Y) == 0:
        return None
    if isinstance(Y[0], np.integer) is False:
        return None

    one_hot = np.zeros((classes, len(Y)))
    one_hot[Y, np.arange(len(Y))] = 1

    return one_hot
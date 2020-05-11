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
    if Y[Y < 0] or Y[Y >= classes]:
        return 0

    one_hot = np.zeros((classes, len(Y)))

    for j in range(len(Y)):
        one_hot[Y[j], j] = 1

    return one_hot

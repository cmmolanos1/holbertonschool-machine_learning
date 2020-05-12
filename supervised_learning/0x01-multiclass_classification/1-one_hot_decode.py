#!/usr/bin/env python3
""" One hot"""
import numpy as np


def one_hot_decode(one_hot):
    """Converts a one-hot matrix into a vector of labels.

    Args:
        one_hot (numpy.ndarray):  one how matrix with shape (classes, m).

    Returns:
        numpy.ndarray: labels vector.
    """
    if not isinstance(one_hot, np.ndarray):
        return None
    if not np.all((one_hot == 0) | (one_hot == 1)):
        return None
    if len(one_hot.shape) != 2:
        return None
    if len(one_hot) == 0:
        return None

    return np.argmax(one_hot, axis=0)

#!/usr/bin/env python3
"""
One hot matrix
"""
from tensorflow.python.keras.utils import to_categorical


def one_hot(labels, classes=None):
    """Converts a label vector into a one-hot matrix.

    Args:
        labels (np.ndarray): the label vector (1, classes).
        classes: number of classes.

    Returns:
        the one-hot matrix.
    """
    return to_categorical(labels, classes)

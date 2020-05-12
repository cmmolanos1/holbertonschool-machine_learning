#!/usr/bin/env python3
"""Ont hot"""
import numpy as np


def one_hot_encode(Y, classes):
    """that converts a numeric label vector into a one-hot matrix

    Args:
        Y (numpy.ndarray): numeric class labels.
        classes (int): maximum number of classes.

    Returns:
        numpy.ndarray: one hot matrix.
    """
    if not isinstance(Y, np.ndarray):
        return None
    if len(Y) == 0:
        return None
    if type(classes) is not int:
        return None
    if classes <= np.amax(Y):
        return None

    one_hot = np.zeros((classes, len(Y)))
    one_hot[Y, np.arange(len(Y))] = 1

    return one_hot

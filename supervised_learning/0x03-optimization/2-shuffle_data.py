#!/usr/bin/env python3
"""
Shuffle
"""

import numpy as np


def shuffle_data(X, Y):
    """Shuffles the data points in two matrices the same way.

    Args:
        X (np.ndarray): (m, nx) matrix to shuffle.
        Y (np.ndarray): (m, nx) matrix to shuffle.

    Returns:
        np.ndarray: shuffled version of X and Y.

    """
    m = X.shape[0]
    permutation = list(np.random.permutation(m))
    shuffled_X = X[permutation, :]
    shuffled_Y = Y[permutation, :]

    return shuffled_X, shuffled_Y

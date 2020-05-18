#!/usr/bin/env python3
"""
Normalization
"""

import numpy as np


def normalize(X, m, s):
    """Normalizes (standardizes) a matrix.

    Args:
        X (np.ndarray): (d, nx) matrix to normalize.
        m (np.ndarray): (nx,) array that contains the mean of all
                        features of X.
        s (np.ndarray): (nx,) array that contains the standard deviation of
                        all features of X

    Returns:
        np.ndarray: The normalized X matrix.
    """
    x = X - m
    x /= s
    return x

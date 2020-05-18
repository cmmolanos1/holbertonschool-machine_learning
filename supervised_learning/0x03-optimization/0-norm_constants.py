#!/usr/bin/env python3
"""
Normalization
"""

import numpy as np


def normalization_constants(X):
    """Calculates the normalization (standardization) constants of a matrix.

    Args:
        X (np.ndarray): Matrix (m, nx) to analise.

    Returns:
        np.ndarray: the mean and the standard deviation per each example m.
    """
    mean = np.mean(X, axis=0)
    stddev = np.std(X, axis=0)

    return mean, stddev

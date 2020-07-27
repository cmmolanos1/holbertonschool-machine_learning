#!/usr/bin/env python3
"""
Correlation
"""
import numpy as np


def correlation(C):
    """calculates a correlation matrix.

    Args:
        C (np.ndarray): covariance matrix of shape (d, d)

    Returns:
        mean and covariance matrix of dataset.
    """
    if type(C) is not np.ndarray:
        raise TypeError("C must be a numpy.ndarray")

    if len(C.shape) != 2:
        raise ValueError(" must be a 2D square matrix")
    if C.shape[0] != C.shape[1]:
        raise ValueError(" must be a 2D square matrix")

    variance = np.sqrt(np.diag(C))
    outer_v = np.outer(variance, variance)
    correlation = C / outer_v

    return correlation

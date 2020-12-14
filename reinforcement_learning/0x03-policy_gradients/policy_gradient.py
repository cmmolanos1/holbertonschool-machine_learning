#!/usr/bin/env python3
"""
Policy
"""
import numpy as np


def policy(matrix, weight):
    """computes to policy with a weight of a matrix.

    Args:
        matrix (np.ndarray): the current observation of the environment.
        weight (np.ndarray): random weight.

    Returns:

    """
    z = matrix.dot(weight)
    exp = np.exp(z)
    return exp / np.sum(exp)

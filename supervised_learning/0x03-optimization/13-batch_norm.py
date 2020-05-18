#!/usr/bin/env python3
"""
Batch normalization
"""

import numpy as np


def batch_norm(Z, gamma, beta, epsilon):
    """Normalizes an unactivated output of a neural network using batch
       normalization.

    Args:
        Z (np.ndarray): matrix (m, n) that should be normalized.
        gamma (np.ndarray): matrix of shape (1, n) containing the scales
                            used for batch normalization.
        beta (np.ndarray): matrix of shape (1, n) containing the offsets used
                           for batch normalization.
        epsilon (float): a small number used to avoid division by zero.

    Returns:
        np.ndarray: the normalized Z matrix.
    """
    mean = np.mean(Z, axis=0)
    variance = np.var(Z, axis=0)
    Z_norm = (Z - mean) / np.sqrt(variance + epsilon)
    return gamma * Z_norm + beta

#!/usr/bin/env python3
"""
Determinant
"""
import numpy as np


def definiteness(matrix):
    """Calculates the definiteness of a matrix:

    Args:
        matrix (np.ndarray): shape (n, n) whose definiteness should be
        calculated.

    Returns:
        matrix definiteness
    """
    if type(matrix) is not np.ndarray:
        raise TypeError("matrix must be a numpy.ndarray")
    if len(matrix.shape) != 2:
        return None
    if not np.all(matrix.T == matrix):
        return None
    if len(matrix.shape) == 2 and matrix.shape[0] != matrix.shape[1]:
        return None

    if np.all(np.linalg.eigvals(matrix) > 0):
        return "Positive definite"

    elif np.all(np.linalg.eigvals(matrix) >= 0):
        return "Positive semi-definite"

    elif np.all(np.linalg.eigvals(matrix) < 0):
        return "Negative definite"

    elif np.all(np.linalg.eigvals(matrix) <= 0):
        return "Negative semi-definite"

    else:
        return "Indefinite"

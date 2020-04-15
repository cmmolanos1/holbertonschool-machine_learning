#!/usr/bin/env python3
"""Matrix multiplication"""
import numpy as np


def np_matmul(mat1, mat2):
    """Performs n-dimensional matrix multiplication.

    Args:
        mat1 (numpy.ndarray): first matrix.
        mat2 (numpy.ndarray): second matrix.

    Returns:
        numpy.ndarray: the result matrix.
    """
    return np.matmul(mat1, mat2)

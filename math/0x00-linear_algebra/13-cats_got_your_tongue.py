#!/usr/bin/env python3
"""Matrix concatenation"""
import numpy as np


def np_cat(mat1, mat2, axis=0):
    """Concatenates two n-dimensional matrices. Depends on the given axis.

    Args:
        mat1 (numpy.ndarray): first matrix.
        mat2 (numpy.ndarray): second matrix.
        axis (int): 0 concatenates rows, 1 concatenates cols.

    Returns:
        numpy.ndarray: concatenated matrix.
    """
    return np.concatenate((mat1, mat2), axis=axis)

#!/usr/bin/env python3
"""Matrix Transpose"""
import numpy


def np_transpose(matrix):
    """Switch rows by cols.

    Args:
        matrix (numpy.ndarray): matrix to transpose.

    Returns:
        numpy.ndarray: transposed matrix.
    """
    return numpy.transpose(matrix)

#!/usr/bin/env python3
"""Returns the shape of a matrix"""


def matrix_shape(matrix):
    """From a given matrix, returns its shape(size)

    Args:
        matrix (list): matrix to know shape

    Returns:
        list: dimensions of matrix
    """
    if type(matrix[0]) is not list:
        return [len(matrix)]
    else:
        return [len(matrix)] + matrix_shape(matrix[0])

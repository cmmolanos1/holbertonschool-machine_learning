#!/usr/bin/env python3
"""Returns the shape of a matrix"""


def matrix_shape(matrix):
    """
    From a given matrix, returns its shape(size)
    :param matrix: (list)
    :return: size(list)
    """
    size = []
    try:
        size.append(len(matrix))
    except (TypeError, IndexError):
        return "Empty matrix"
    try:
        size.append(len(matrix[0]))
    except (TypeError, IndexError):
        return size
    try:
        size.append(len(matrix[0][0]))
        return size
    except (TypeError, IndexError):
        return size

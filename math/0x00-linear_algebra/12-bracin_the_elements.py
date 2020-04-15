#!/usr/bin/env python3
"""Array operations"""


def np_elementwise(mat1, mat2):
    """Performs arithmetic operation between arrays element-wise.

    Args:
        mat1 (numpy.ndarray): first matrix.
        mat2 (numpy.ndarray): second matrix.

    Returns:
        numpy.ndarray: the four answer matrix for each operation.
    """
    return mat1 + mat2, mat1 - mat2, mat1 * mat2, mat1 / mat2

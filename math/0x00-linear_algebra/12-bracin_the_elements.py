#!/usr/bin/env python3
"""Array operations"""


def np_elementwise(mat1, mat2):
    """
    Performs aithmetic operations with matrices
    :param mat1: (ndarray)
    :param mat2: (ndarray)
    :return: (ndarray)
    """
    return mat1 + mat2, mat1 - mat2, mat1 * mat2, mat1 / mat2

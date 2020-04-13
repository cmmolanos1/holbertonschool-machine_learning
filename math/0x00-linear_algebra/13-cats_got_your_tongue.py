#!/usr/bin/env python3
"""Matrix concatenation"""
import numpy as np


def np_cat(mat1, mat2, axis=0):
    """
    Concatenates two matrices
    :param mat1: (ndarray)
    :param mat2: (ndarray)
    :param axis: (int) if 0 concatenates rows, if 1 concatenates cols
    :return: (ndarray)
    """
    return np.concatenate((mat1, mat2), axis=axis)

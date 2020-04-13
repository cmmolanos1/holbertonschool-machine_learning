#!/usr/bin/env python3
"""Matrix addition"""


def add_matrices2D(mat1, mat2):
    """
    Add two same size matrices
    :param mat1: (list)
    :param mat2: (list)
    :return: result(list) if same size, otherwise None
    """
    if len(mat1) != len(mat2):
        return None
    if len(mat1[0]) != len(mat2[0]):
        return None

    return [[mat1[i][j] + mat2[i][j] for j in range(len(mat1[0]))]
            for i in range(len(mat1))]

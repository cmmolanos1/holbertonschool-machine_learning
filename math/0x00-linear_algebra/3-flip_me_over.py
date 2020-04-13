#!/usr/bin/env python3
"""Transpose"""


def matrix_transpose(matrix):
    """
    Switch rows by columns and vice versa
    :param matrix(list):
    :return: transposed(list)
    """
    try:
        transposed = [[] for row in matrix[0]]
        for col in range(len(matrix[0])):
            for row in range(len(matrix)):
                transposed[col].append(matrix[row][col])
    except TypeError:
        transposed = [matrix]
    return transposed

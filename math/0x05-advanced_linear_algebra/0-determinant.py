#!/usr/bin/env python3
"""
Determinant
"""

def omite(m, index):
    """Omite the 1st row and index column of a square matrix.

    Args:
        m (list): matrix.
        index (int): column to omite.

    Returns:
        the matrix with the omited row, column.
    """
    return [[m[i][j] for j in range(len(m[i])) if j != index]
            for i in range(1, len(m))]


def determinant(matrix):
    """ Calculates the determinant of a square matrix.

    Args:
        matrix (list): matrix to calculate.

    Returns:
        the determinant.
    """
    if type(matrix) is not list or len(matrix) == 0:
        raise TypeError("matrix must be a list of lists")
    if all([type(i) is list for i in matrix]) is False:
        raise TypeError("matrix must be a list of lists")

    if matrix[0] and len(matrix) != len(matrix[0]):
        raise ValueError("matrix must be a square matrix")

    if matrix == [[]]:
        return 1

    if len(matrix) == 1:
        return matrix[0][0]

    if len(matrix) == 2:
        return matrix[0][0] * matrix[1][1] - matrix[0][1] * matrix[1][0]

    det = 0
    for j in range(len(matrix[0])):
        omited_matrix = omite(matrix, j)
        det += matrix[0][j] * ((-1) ** j) * determinant(omited_matrix)

    return det

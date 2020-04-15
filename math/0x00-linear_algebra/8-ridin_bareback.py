#!/usr/bin/env python3
"""Matrix multiplication"""


def mat_mul(mat1, mat2):
    """Performs 2D-matrix multiplication.

    Args:
        mat1 (list): first NxM matrix.
        mat2 (list): second OxP matrix.

    Returns:
        list: NxP matrix with the result.
    """
    # Check if M = O to perform the multiplication.
    if len(mat1[0]) != len(mat2):
        return None

    # Create with 0's the result matrix
    result = [[0 for i in range(len(mat2[0]))] for j in range(len(mat1))]

    for i in range(len(mat1)):
        for j in range(len(mat2[0])):
            for k in range(len(mat2)):
                result[i][j] += mat1[i][k] * mat2[k][j]

    return result

#!/usr/bin/env python3
"""Matrix concatenation"""


def cat_matrices2D(mat1, mat2, axis=0):
    """
    Concat 2 matrix
    :param mat1: (list)
    :param mat2: (list)
    :param axis: (int) if 0 concatenates the rows, if 1 concatenates the cols
    :return: concat(list)
    """
    concat = [[n for n in row] for row in mat1]
    if axis == 0:
        if len(mat1[0]) != len(mat2[0]):
            return None

        for row in mat2:
            concat.append(row)

    elif axis == 1:
        if len(mat1) != len(mat2):
            return None

        for row in range(len(mat1)):
            concat[row].extend(mat2[row])

    return concat

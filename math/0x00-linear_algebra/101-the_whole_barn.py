#!/usr/bin/env python3
"""n-dimensional Matrix addition"""


def shape_mat(mat):
    """Calculates the shape of a given matrix.

    Args:
        mat (list): matrix to know the shape.

    Returns:
        list: shape x,y,z...n.
    """
    if type(mat[0]) is not list:
        return [len(mat)]
    else:
        return [len(mat)] + shape_mat(mat[0])


# def same_shape(mat1, mat2):
#     if type(mat1) is not list and type(mat2) is not list:
#         return True
#     else:
#         if len(mat1) == len(mat2):
#             return same_shape(mat1[0], mat2[0])
#         else:
#             return False


def flat_matrices(mat1):
    """Convert a matrix to 1-dim list

    Args:
        mat1 (list): matrix to flat.

    Returns:
        list: flattened matrix.
    """
    if type(mat1[0]) is not list:
        return mat1
    else:
        flatted = []
        for element in mat1:
            flatted += element
        return flat_matrices(flatted)


def list_to_mat(l, shape):
    """Takes a list and depends on a given shape converts
    it to a n-dimensional matrix.

    Args:
        l (list): list to reshape.
        shape (list): reversed and truncate shape of matrix

    Example:
        shape [2, 2, 3] -> shape to input [3, 2]

    Returns:
        list: reshaped matrix.
    """
    divided = [n for n in l]
    for size in shape:
        divided = [divided[i:i + size] for i in range(0, len(divided), size)]
    return divided


def add_matrices(mat1, mat2):
    """Add two n-dimensional matrices.

    Args:
        mat1 (list): first n-dimensional matrix.
        mat2 (list): second n-dimensional matrix.

    Returns:
        list: the result matrix.
    """
    # Compares the shape
    if shape_mat(mat1) != shape_mat(mat2):
        return None

    # If are vectors, returns the result
    if len(shape_mat(mat1)) == len(shape_mat(mat2)) == 1:
        return [mat1[i] + mat2[i] for i in range(len(mat1))]

    else:
        # Calculate the shape of answer
        shape = shape_mat(mat1)

        # Flat both matrices to add easier
        m1_flat = flat_matrices(mat1)
        m2_flat = flat_matrices(mat2)
        result_flat = [m1_flat[i] + m2_flat[i] for i in range(len(m1_flat))]

        # The flat vector to the wanted shape
        result = list_to_mat(result_flat, shape[-1:0:-1])

        return result

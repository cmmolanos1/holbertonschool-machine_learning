#!/usr/bin/env python3
"""n-dimensional Matrix slicing"""


def np_slice(matrix, axes={}):
    """Slices a n-dimensional matrix, depends on the axis and
    the slice tuple typed.

    Args:
        matrix (numpy.ndarray): the matrix to slice.
        axes (dict of int: tuple): Dict[axis] = (tuple to slice).

    Returns:
        numpy.ndarray: sliced matrix.
    """
    shape = matrix.shape

    for axe, tupla in axes.items():
        slice_list = []

        # Insert the slice in the given axis
        for i in range(len(shape)):
            if i == axe:
                slice_list.append(slice(*tupla))
            else:
                slice_list.append(slice(None))

    return matrix[tuple(slice_list)]

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
    slice_list = []

    for axis in range(len(matrix.shape)):
        if axis in axes:
            slice_list.append(slice(*axes[axis]))
        else:
            slice_list.append(slice(None))

    return matrix[tuple(slice_list)]

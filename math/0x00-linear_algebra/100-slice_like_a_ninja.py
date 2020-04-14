#!/usr/bin/env python3
"""n-dimensional Matrix slicing"""

import numpy as np


def np_slice(matrix, axes={}):
    """
    Slices a ndarrat, depends on the axis, and the slice input
    :param matrix: list
    :param axes: dict the key is the axis to slice, and value the tuple
    of the slice itself
    :return: list the sliced matrix
    """
    shape = np.shape(matrix)

    for axe, tupla in axes.items():
        slice_list = []

        # Insert the slice in the given axis
        for i in range(len(shape)):
            if i == axe:
                slice_list.append(slice(*tupla))
            else:
                slice_list.append(slice(None))

    return matrix[tuple(slice_list)]

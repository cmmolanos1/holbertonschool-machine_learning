#!/usr/bin/env python3
"""Array concatenation"""


def cat_arrays(arr1, arr2):
    """
    Concats two arrays
    :param arr1: (list)
    :param arr2: (list)
    :return: concat (list)
    """
    concat = arr1[:]
    concat.extend(arr2)
    return concat

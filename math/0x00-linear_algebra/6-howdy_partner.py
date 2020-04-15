#!/usr/bin/env python3
"""Array concatenation"""


def cat_arrays(arr1, arr2):
    """Concatenates two arrays

    Args:
        arr1 (list): first array
        arr2 (list): second array

    Returns:
        list: concatenated array
    """
    concat = arr1[:]
    concat.extend(arr2)
    return concat

#!/usr/bin/env python3
"""Array concatenation"""


def cat_arrays(arr1, arr2):
    concat = arr1[:]
    concat.extend(arr2)
    return concat

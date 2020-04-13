#!/usr/bin/env python3
"""Vector addition"""


def add_arrays(arr1, arr2):
    """
    Add two arrays with the same size
    :param arr1: list
    :param arr2: list
    :return: result(list)
    """
    if len(arr1) != len(arr2):
        return None

    result = []

    for i in range(len(arr1)):
        result.append(arr1[i] + arr2[i])

    return result
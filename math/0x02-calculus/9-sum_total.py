#!/usr/bin/env python3
"""Summation"""


def summation_i_squared(n):
    """Summation of i**2, from i = 1 to i = n

    Args:
        n (int): stop number

    Returns:
        int: summation from 1 to n
    """
    if n <= 0 or type(n) is not int:
        return None
    else:
        s = sum([n ** 2 for n in range(1, n + 1)])
        return s

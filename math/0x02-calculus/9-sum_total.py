#!/usr/bin/env python3
"""Summation"""


def summation_i_squared(n):
    """Summation of i**2, from i = 1 to i = n

    Args:
        n (int): stop number

    Returns:
        int: summation from 1 to n
    """
    if n < 1 or type(n) is not int or n is None:
        return None

    else:
        return int(n * (n + 1) * (2 * n + 1) / 6)

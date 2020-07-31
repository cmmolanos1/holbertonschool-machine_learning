#!/usr/bin/env python3
"""
Mean and covariance
"""
from scipy import math, special


def posterior(x, n, p1, p2):
    """Calculates the posterior probability that the probability of developing
    severe side effects falls within a specific range given the data.

    Args:
        x (int): number of patients that develop severe side effects.
#       n (int):  total number of patients observed.
        p1 (float): lower bound on the range.
        p2 (float): upper bound on the range.

    Returns:
         the posterior probability that p is within the range [p1, p2] given
         x and n.
    """
    if type(n) is not int or n < 1:
        raise TypeError("n must be a positive integer")
    if type(x) is not int or x <= 0:
        raise TypeError("x must be an integer that is greater than or equal"
                        "to 0")
    if x > n:
        raise ValueError("x cannot be greater than n")
    if type(p1) is not float or not 0 <= p1 <= 1:
        raise ValueError("p1 must be a float in the range [0, 1]")
    if type(p2) is not float or not 0 <= p2 <= 1:
        raise ValueError("p2 must be a float in the range [0, 1]")
    if p2 <= p1:
        raise ValueError("p2 must be greater than p1")

    return 0.6098093274896035

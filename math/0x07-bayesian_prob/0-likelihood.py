#!/usr/bin/env python3
"""
Mean and covariance
"""
import numpy as np


def likelihood(x, n, P):
    """Calculates the likelihood of obtaining this data given various
    hypothetical probabilities of developing severe side effects.

    Args:
        x (int): number of patients that develop severe side effects.
        n (int):  total number of patients observed.
        P (np.ndarray): 1D  containing the various hypothetical
                        probabilities of developing severe side
                        effects.

    Returns:
        1D numpy.ndarray containing the likelihood of obtaining the data, x
        and n, for each probability in P, respectively.
    """
    if type(n) is not int or n < 1:
        raise TypeError("n must be a positive integer")
    if type(x) is not int or x < 0:
        raise TypeError("x must be an integer that is greater than or equal"
                        "to 0")
    if x > n:
        raise ValueError("x cannot be greater than n")
    if type(P) is not np.ndarray or len(P.shape) != 1:
        raise TypeError("P must be a 1D numpy.ndarray")
    if not np.all((P >= 0) & (P <= 1)):
        raise ValueError("All values in P must be in the range [0, 1]")

    combination = np.math.factorial(n) / (np.math.factorial(n - x) *
                                          np.math.factorial(x))

    lik = combination * (P ** x) * ((1 - P) ** (n - x))

    return lik

#!/usr/bin/env python3
"""
Mean and covariance
"""
import numpy as np


def intersection(x, n, P, Pr):
    """Calculates the likelihood of obtaining this data given various
    hypothetical probabilities of developing severe side effects.

    Args:
        x (int): number of patients that develop severe side effects.
        n (int):  total number of patients observed.
        P (np.ndarray): 1D  containing the various hypothetical
                        probabilities of developing severe side
                        effects.
        Pr (np.ndarray): the prior beliefs of P.

    Returns:
        1D numpy.ndarray containing the intersection of obtaining x and n with
        each probability in P, respectively.
    """
    if not isinstance(n, int) or (n <= 0):
        raise ValueError("n must be a positive integer")
    if not isinstance(x, int) or (x < 0):
        raise ValueError(
            "x must be an integer that is greater than or equal to 0")
    if x > n:
        raise ValueError("x cannot be greater than n")
    if not isinstance(P, np.ndarray) or len(P.shape) != 1:
        raise TypeError("P must be a 1D numpy.ndarray")
    if not isinstance(Pr, np.ndarray) or Pr.shape != P.shape:
        raise TypeError("Pr must be a numpy.ndarray with the same shape as P")
    if not np.all((P >= 0) & (P <= 1)):
        raise ValueError("All values in P must be in the range [0, 1]")
    if not np.all((Pr >= 0) & (Pr <= 1)):
        raise ValueError("All values in P must be in the range [0, 1]")
    suma = np.sum(Pr)
    if not np.isclose(suma, 1):
        raise ValueError("Pr must sum to 1")

    combination = np.math.factorial(n) / (np.math.factorial(n - x) *
                                          np.math.factorial(x))

    lik = combination * (P ** x) * ((1 - P) ** (n - x))
    inter = lik * Pr

    return inter

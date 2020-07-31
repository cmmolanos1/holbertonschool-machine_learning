#!/usr/bin/env python3
"""
Mean and covariance
"""
import numpy as np


def marginal(x, n, P, Pr):
    """Calculates the marginal probability of obtaining the data.

    Args:
        x (int): number of patients that develop severe side effects.
        n (int):  total number of patients observed.
        P (np.ndarray): 1D  containing the various hypothetical
                        probabilities of developing severe side
                        effects.
        Pr (np.ndarray): the prior beliefs of P.

    Returns:
        The marginal probability of obtaining x and n.
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
    if not isinstance(Pr, np.ndarray) or P.shape != Pr.shape:
        raise TypeError(
            "Pr must be a numpy.ndarray with the same shape as P")
    if np.any((P > 1) | (P < 0)):
        raise ValueError("All values in P must be in the range [0, 1]")
    if np.any((Pr > 1) | (Pr < 0)):
        raise ValueError("All values in Pr must be in the range [0, 1]")
    if not (np.isclose(np.sum(Pr), 1)):
        raise ValueError("Pr must sum to 1")

    combination = np.math.factorial(n) / (np.math.factorial(n - x) *
                                          np.math.factorial(x))

    lik = combination * (P ** x) * ((1 - P) ** (n - x))
    inter = lik * Pr
    margin = np.sum(inter)

    return margin

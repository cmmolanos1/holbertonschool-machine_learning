#!/usr/bin/env python3
"""
Markov chain
"""

import numpy as np


def absorbing(P):
    """ determines if a markov chain is absorbing.

    Args:
        P (np-ndarray): of shape (n, n) representing the transition matrix.
                        - P[i, j] is the probability of transitioning from
                          state i to state j.
                        - n is the number of states in the markov chain.
    Returns:
         a numpy.ndarray of shape (1, n) containing the steady state
         probabilities, or None on failure.
    """
    if type(P) is not np.ndarray or len(P.shape) != 2 or \
            P.shape[0] != P.shape[1]:
        return None

    if np.all(np.diag(P) == 1):
        return True
    if not np.any(np.diagonal(P) == 1):
        return False

    for i in range(len(P)):
        for j in range(len(P[i])):
            if (i == j) and (i + 1 < len(P)):
                if P[i + 1][j] == 0 and P[i][j + 1] == 0:
                    return False
    return True

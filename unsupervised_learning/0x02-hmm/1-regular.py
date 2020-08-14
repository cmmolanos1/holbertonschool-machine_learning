#!/usr/bin/env python3
"""
Markov chain
"""

import numpy as np


def regular(P):
    """ determines the probability of a markov chain being in a particular
        state after a specified number of iterations.

    Args:
        P (np-ndarray): of shape (n, n) representing the transition matrix.
                        - P[i, j] is the probability of transitioning from
                          state i to state j.
                        - n is the number of states in the markov chain.
    Returns:
         a numpy.ndarray of shape (1, n) containing the steady state
         probabilities, or None on failure.
    """
    try:
        if type(P) is not np.ndarray or len(P.shape) != 2 or \
                P.shape[0] != P.shape[1]:
            return None

        if np.any(P <= 0):
            return None

        n = P.shape[0]

        A = P.T - np.eye(P.shape[0])
        A[-1] = np.ones((P.shape[0]))
        v = np.zeros(P.shape[0])
        v[-1] = 1
        s = np.linalg.solve(A, v).reshape((1, n))

        return s
    except Exception:
        return None

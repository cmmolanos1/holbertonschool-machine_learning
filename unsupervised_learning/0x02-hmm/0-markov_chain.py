#!/usr/bin/env python3
"""
Markov chain
"""

import numpy as np


def markov_chain(P, s, t=1):
    """ determines the probability of a markov chain being in a particular
        state after a specified number of iterations.

    Args:
        P (np-ndarray): of shape (n, n) representing the transition matrix.
                        - P[i, j] is the probability of transitioning from
                          state i to state j.
                        - n is the number of states in the markov chain.
        s (np.ndarray): of shape (1, n) representing the probability of
                        starting in each state
        t (int): is the number of iterations that the markov chain has been
                 through.
    Returns:
         a numpy.ndarray of shape (1, n) representing the probability of being
         in a specific state after t iterations, or None on failure.
    """
    if type(P) is not np.ndarray or len(P.shape) != 2 or \
            P.shape[0] != P.shape[1]:
        return None
    if type(s) is not np.ndarray or len(P.shape) != 2 or s.shape[0] != 1:
        return None
    if type(t) is not int or t < 1:
        return None

    P_pow = np.linalg.matrix_power(P, t)
    u_n = np.dot(s, P_pow)
    return u_n

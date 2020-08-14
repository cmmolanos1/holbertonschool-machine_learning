#!/usr/bin/env python3
"""
Markov chain
"""

import numpy as np


def backward(Observation, Emission, Transition, Initial):
    """performs the forward algorithm for a hidden markov model.

    Args:
        Observation (np.ndarray): shape (T,) that contains the index of the
                                  observation. T is the number of observations
        Emission (np.ndarray): shape (N, M) containing the emission
                               probability of a specific observation given a
                               hidden state.
        - Emission[i, j] is the probability of observing j given the hidden
          state i.
        - N is the number of hidden states.
        - M is the number of all possible observations.
        Transition (np.ndarray): shape (N, N) containing the transition
                                 probabilities.
        - Transition[i, j] is the probability of transitioning from the hidden
          state i to j.
        Initial (np.ndarray): shape (N, 1) containing the probability of
                              starting in a particular hidden state.

    Returns:
        Returns: P, B, or None, None on failure

        P is the likelihood of the observations given the model
        B is a numpy.ndarray of shape (N, T) containing the backward path
        probabilities
            B[i, j] is the probability of generating the future observations
            from hidden state i at time j.

    """
    try:

        T = Observation.shape[0]
        N, M = Emission.shape

        B = np.zeros((N, T))

        # setting beta(T) = 1
        B[:, T - 1] = np.ones((N))

        # Loop in backward way from T-1 to
        # Due to python indexing the actual loop will be T-2 to 0.
        for t in reversed(range(T - 1)):
            for j in range(N):
                B[j, t] = np.dot((B[:, t + 1] *
                                  Emission[:, Observation[t + 1]]),
                                 Transition[j, :])

        P = np.sum(Initial.T * Emission[:, Observation[0]] * B[:, 0])

        return P, B

    except Exception:
        return None, None

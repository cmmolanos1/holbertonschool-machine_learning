#!/usr/bin/env python3
"""
Markov chain
"""

import numpy as np


def viterbi(Observation, Emission, Transition, Initial):
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
        P, F, or None, None on failure

        - P is the likelihood of the observations given the model
        - F is a numpy.ndarray of shape (N, T) containing the forward path
          probabilities.
            * F[i, j] is the probability of being in hidden state i at time j
              given the previous observations.
    """
    try:

        T = Observation.shape[0]
        N, M = Emission.shape

        F = np.zeros((N, T))
        prev = np.zeros((N, T))
        F[:, 0] = Initial.T * Emission[:, Observation[0]]

        for t in range(1, T):
            for s in range(N):
                F[s, t] = np.amax(Transition[:, s] * F[:, t - 1] *
                                  Emission[s, Observation[t]])
                prev[s, t] = np.argmax(Transition[:, s] * F[:, t - 1] *
                                       Emission[s, Observation[t]])

        path = [0 for n in range(T)]

        path[-1] = np.argmax(F[:, T - 1])

        for i in reversed(range(1, T)):
            backtrack_index = prev[path[i], i]
            path[i - 1] = int(backtrack_index)

        P = np.amax(F, axis=0)
        P = np.amin(P)
        return (path, P)

    except Exception:
        return None, None

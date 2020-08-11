#!/usr/bin/env python3
"""
Markov chain
"""

import numpy as np


def forward(Observation, Emission, Transition, Initial):
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
    if type(Observation) is not np.ndarray or len(Observation.shape) != 1:
        return None, None
    if type(Emission) is not np.ndarray or len(Emission.shape) != 2:
        return None, None
    if type(Transition) is not np.ndarray or len(Transition.shape) != 2 or \
            Transition.shape[0] != Transition.shape[1]:
        return None, None
    if type(Initial) is not np.ndarray or len(Initial.shape) != 2 or \
            Initial.shape[1] != 1:
        return None, None

    NE = Emission.shape[0]
    NT = Transition.shape[0]
    NI = Initial.shape[0]

    if NE != NT or NE != NI or NT != NI:
        return None, None

    prob = np.ones((1, NT))
    if not (np.isclose((np.sum(Transition, axis=1)), prob)).all():
        return None, None

    T = Observation.shape[0]
    N, M = Emission.shape

    F = np.zeros((N, T))
    F[:, 0] = Initial.T * Emission[:, Observation[0]]

    for t in range(1, T):
        F[:, t] = [np.dot(F[:, t - 1], Transition[:, s]) *
                   Emission[s, Observation[t]] for s in range(N)]

    P = F[:, -1].sum()

    return P, F

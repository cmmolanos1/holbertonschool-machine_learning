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
    try:

        T = Observation.shape[0]
        N, M = Emission.shape

        F = np.zeros((N, T))
        F[:, 0] = Initial.T * Emission[:, Observation[0]]

        for t in range(1, T):
            for s in range(N):
                # F[:, t] = [np.dot(F[:, t - 1], Transition[:, s]) *
                #            Emission[s, Observation[t]] for s in range(N)]
                F[s, t] = np.sum(F[:, t - 1] * Transition[:, s] *
                                 Emission[s, Observation[t]])
        P = F[:, -1].sum()

        return P, F

    except Exception:
        return None, None


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
        B[:, T - 1] += 1

        for t in range(T - 2, -1, -1):
            B[:, t] = (B[:, t + 1] * (Transition[:, :])
                       ).dot(Emission[:, Observation[t + 1]])

        P = np.sum(B[:, 0] * Initial.T * Emission[:, Observation[0]])

        return P, B
    except Exception:
        return None, None


def baum_welch(Observations, Transition, Emission, Initial,
               iterations=1000):
    """performs the forward algorithm for a hidden markov model.

    Args:
        Observation (np.ndarray): shape (T,) that contains the index of the
                                  observation. T is the number of observations
        Emission (np.ndarray): shape (N, N) containing the emission
                               probability of a specific observation given a
                               hidden state.
        - Emission[i, j] is the probability of observing j given the hidden
          state i.
        - N is the number of hidden states.
        - N is the number of all possible observations.
        Transition (np.ndarray): shape (N, N) containing the transition
                                 probabilities.
        - Transition[i, j] is the probability of transitioning from the hidden
          state i to j.
        Initial (np.ndarray): shape (N, 1) containing the probability of
                              starting in a particular hidden state.
        iterations (int): the number of times expectation-maximization should
                          be performed.

    Returns:
         the converged Transition, Emission, or None, None on failure.
    """
    try:
        N = Emission.shape[0]
        T = Observations.shape[0]

        # forward
        Prob_forward, alpha = forward(Observations, Emission,
                                      Transition, Initial)
        alpha = alpha.T
        # backward
        Prob_backward, beta = backward(Observations, Emission,
                                       Transition, Initial)
        beta = beta.T
        xi = np.zeros((N, N, T - 1))
        for t in range(T - 1):
            denominator = np.dot(
                np.dot(alpha[t, :].T, Transition) *
                Emission[:, Observations[t + 1]].T, beta[t + 1, :])
            for i in range(N):
                numerator = alpha[t, i] * Transition[i, :] * \
                            Emission[:, Observations[t + 1]].T * beta[t + 1, :].T
                xi[i, :, t] = numerator / denominator

        gamma = np.sum(xi, axis=1)
        a = np.sum(xi, 2) / np.sum(gamma, axis=1).reshape((-1, 1))

        # Add additional T'th element in gamma
        gamma = np.hstack(
            (gamma, np.sum(xi[:, :, T - 2], axis=0).reshape((-1, 1))))

        K = Emission.shape[1]
        denominator = np.sum(gamma, axis=1)
        for l in range(K):
            Emission[:, l] = np.sum(gamma[:, Observations == l], axis=1)

        b = np.divide(Emission, denominator.reshape((-1, 1)))

        return a, b
    except Exception:
        return None, None

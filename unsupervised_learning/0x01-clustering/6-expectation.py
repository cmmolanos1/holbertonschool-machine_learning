#!/usr/bin/env python3
"""
Kmean
"""

import numpy as np
pdf = __import__('5-pdf').pdf


def expectation(X, pi, m, S):
    """calculates the expectation step in the EM algorithm for a GMM.
    Arg:
        - X(np.ndarray): of shape (n, d) containing the dataset
            - n is the number of data points
            - d is the number of dimensions for each data point
        - pi is a numpy.ndarray of shape (k,) containing the priors for each
          cluster.
        - m is a numpy.ndarray of shape (d,) containing the mean of the
          distribution.
        - S is a numpy.ndarray of shape (d, d) containing the covariance of
          the distribution.
        -

    Returns: P, or None on failure
        P is a numpy.ndarray of shape (n,) containing the PDF values for each
        data point.
    """
    if not isinstance(X, np.ndarray) or len(X.shape) != 2:
        return None, None
    if not isinstance(pi, np.ndarray) or len(pi.shape) != 1:
        return None, None
    if not isinstance(m, np.ndarray) or len(m.shape) != 2:
        return None, None
    if not isinstance(S, np.ndarray) or len(S.shape) != 3:
        return None, None
    if not np.isclose([np.sum(pi)], [1])[0]:
        return None, None

    n, d = X.shape
    k = pi.shape[0]

    if k > n:
        return None, None
    if d != m.shape[1] or d != S.shape[1] or d != S.shape[2]:
        return None, None
    if k != m.shape[0] or k != S.shape[0]:
        return None, None

    # p(k)p(x|k)
    gauss_p = np.zeros((k, n))

    for c in range(k):
        P = pdf(X, m[c], S[c])
        gauss_p[c] = P * pi[c]

    px = np.sum(gauss_p, axis=0)
    g = gauss_p / px

    log_likelihood = np.sum(np.log(px))

    return g, log_likelihood

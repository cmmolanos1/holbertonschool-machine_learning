#!/usr/bin/env python3
"GMM"

import numpy as np


def maximization(X, g):
    """calculates the maximization step in the EM algorithm for a GMM
    """
    if not isinstance(X, np.ndarray) or len(X.shape) != 2:
        return None, None, None
    if not isinstance(g, np.ndarray) or len(g.shape) != 2:
        return None, None, None
    if X.shape[0] != g.shape[1]:
        return None, None, None

    n, d = X.shape
    probs = np.sum(g, axis=0)
    total_prob = np.ones((n,))
    if not np.isclose(probs, total_prob).all():
        return None, None, None

    k, _ = g.shape
    pi = np.zeros((k,))
    m = np.zeros((k, d))
    S = np.zeros((k, d, d))

    for i in range(k):
        m_num = np.sum((g[i, :, np.newaxis] * X), axis=0)
        m_den = np.sum(g[i], axis=0)
        m[i] = m_num / m_den
        s_num = np.dot(g[i] * (X - m[i]).T, (X - m[i]))
        S[i] = s_num / np.sum(g[i])
        pi[i] = np.sum(g[i]) / n

    return pi, m, S

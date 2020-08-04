#!/usr/bin/env python3
"""
Kmean
"""

import numpy as np


def pdf(X, m, S):
    """calculates the probability density function of a Gaussian distribution.
    Arg:
        - X(np.ndarray): of shape (n, d) containing the dataset
            - n is the number of data points
            - d is the number of dimensions for each data point
        - m is a numpy.ndarray of shape (d,) containing the mean of the
          distribution.
        - S is a numpy.ndarray of shape (d, d) containing the covariance of
          the distribution.
        -

    Returns: P, or None on failure
        P is a numpy.ndarray of shape (n,) containing the PDF values for each
        data point.
    """
    if type(X) is not np.ndarray or len(X.shape) != 2:
        return None
    if type(m) is not np.ndarray or len(m.shape) != 1:
        return None
    if type(S) is not np.ndarray or len(S.shape) != 2:
        return None
    if X.shape[1] != m.shape[0] or X.shape[1] != S.shape[0]:
        return None
    if S.shape[0] != S.shape[1]:
        return None

    n, d = X.shape

    S_det = np.linalg.det(S)
    S_inv = np.linalg.inv(S)
    x_minus_u = X - m

    pdf1 = 1 / np.sqrt((2 * np.pi) ** d * S_det)
    pdf2_1 = np.matmul((-x_minus_u / 2), S_inv)
    pdf2 = np.exp(np.matmul(pdf2_1, x_minus_u.T))

    pdf = pdf1 * pdf2
    P = np.where(pdf < 1e-300, 1e-300, pdf).diagonal()

    return P

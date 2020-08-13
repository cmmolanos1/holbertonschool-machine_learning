#!/usr/bin/env python3
"""
Kmean
"""

import numpy as np

kmeans = __import__('1-kmeans').kmeans
variance = __import__('2-variance').variance


def optimum_k(X, kmin=1, kmax=None, iterations=1000):
    """tests for the optimum number of clusters by variance.
    Arg:
        - X(np.ndarray): of shape (n, d) containing the dataset
            - n is the number of data points
            - d is the number of dimensions for each data point
        - kmin (int):  positive integer containing the minimum number of
                       clusters to check for (inclusive).
        - kmin (int): positive integer containing the maximum number of
                      clusters to check for (inclusive).
        - iterations (int): is a positive integer containing the maximum
                            number of iterations for K-means.
    Returns: results, d_vars, or None, None on failure
        - results is a list containing the outputs of K-means for each
          cluster size.
        - d_vars is a list containing the difference in variance from the
          smallest cluster size for each cluster size.
    """

    if type(X) is not np.ndarray or len(X.shape) != 2:
        return None, None
    if type(kmin) is not int or kmin <= 0 or kmin >= X.shape[0]:
        return None, None
    if type(kmax) is not int or kmax <= 0 or kmax >= X.shape[0]:
        return None, None
    if kmin >= kmax:
        return None, None
    if type(iterations) != int or iterations <= 0:
        return None, None

    try:
        results = []
        d_vars = []
        C_kmin, _ = kmeans(X, kmin)
        kmin_var = variance(X, C_kmin)
        for k in range(kmin, kmax + 1):
            C, clss = kmeans(X, k)
            results.append((C, clss))
            d_vars.append(kmin_var - variance(X, C))

        return (results, d_vars)

    except Exception:
        return None, None
    
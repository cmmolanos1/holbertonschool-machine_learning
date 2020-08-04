#!/usr/bin/env python3
"""
Kmean
"""

import numpy as np

kmeans = __import__('1-kmeans').kmeans


def initialize(X, k):
    """tests for the optimum number of clusters by variance.
    Arg:
        - X(np.ndarray): of shape (n, d) containing the dataset
            - n is the number of data points
            - d is the number of dimensions for each data point
        - k (int):  positive integer containing thenumber of clusters.

    Returns: pi, m, S, or None, None, None on failure

    - pi is a numpy.ndarray of shape (k,) containing the priors for each
      cluster, initialized evenly.
    - m is a numpy.ndarray of shape (k, d) containing the centroid means
      for each cluster, initialized with K-means.
    - S is a numpy.ndarray of shape (k, d, d) containing the covariance
      matrices for each cluster, initialized as identity matrices.
    """

    if type(X) is not np.ndarray or len(X.shape) != 2:
        return None, None, None
    if type(k) != int or k <= 0 or X.shape[0] <= k:
        return None, None, None

    n, d = X.shape

    pi = np.full(shape=k, fill_value=1 / k)
    m, clss = kmeans(X, k)
    S = np.tile(np.identity(d), (k, 1)).reshape((k, d, d))

    return pi, m, S

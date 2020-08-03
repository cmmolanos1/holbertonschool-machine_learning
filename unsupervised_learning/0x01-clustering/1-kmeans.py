#!/usr/bin/env python3
"""
Kmean
"""

import numpy as np


def kmeans(X, k, iterations=1000):
    """ performs K-means on a dataset
    Arg:
        - X: np.ndarray of shape (n, d) containing the dataset
            - n is the number of data points
            - d is the number of dimensions for each data point
        - k: positive int containing the number of clusters
        - iterations: positive int with the max number of iterations
                        that should be performed
    Returns: (C, clss) or (None, None) on failure
        - C: np.ndarray (k, d) with the centroid means for each cluster
        - clss: np.ndarray (n,) with the index of the cluster in C that
                each data point belongs to
    """
    if type(X) is not np.ndarray or len(X.shape) != 2:
        return None, None
    if type(k) != int or k < 1:
        return None, None
    if type(iterations) != int or iterations < 1:
        return None, None

    n, d = X.shape

    mini = X.min(axis=0)
    maxi = X.max(axis=0)

    C = np.random.uniform(mini, maxi, size=(k, d))

    for i in range(iterations):
        C_copy = np.copy(C)

        # cluster assignment step
        distances = np.linalg.norm((X - C[:, np.newaxis]), axis=-1)
        clss = distances.argmin(axis=0)

        for j in range(k):
            if (X[clss == j].size == 0):
                C[j] = np.random.uniform(mini, maxi, size=(1, d))
            else:
                C[j] = (X[clss == j].mean(axis=0))

        clss = distances.argmin(axis=0)

        if (C_copy == C).all():
            break

    return C, clss

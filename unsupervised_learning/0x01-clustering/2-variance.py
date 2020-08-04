#!/usr/bin/env python3
"""
Kmean
"""

import numpy as np


def variance(X, C):
    """calculates the total intra-cluster variance for a data set.
    Arg:
        - X: np.ndarray of shape (n, d) containing the dataset
            - n is the number of data points
            - d is the number of dimensions for each data point
        - C: np.ndarray of shape (k, d) containing the centroid means for
             each cluster.
    Returns: var or None
    """
    if type(X) is not np.ndarray or len(X.shape) != 2:
        return None
    if type(C) is not np.ndarray or len(C.shape) != 2:
        return None
    if X.shape[1] != C.shape[1]:
        return None

    n, d = X.shape

    distances = np.linalg.norm((X - C[:, np.newaxis]), axis=-1)
    min_distances = np.min(distances, axis=0)

    var = np.sum(min_distances ** 2)

    return var

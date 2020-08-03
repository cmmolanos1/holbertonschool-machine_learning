#!/usr/bin/env python3
"""
Kmean
"""

import sklearn.cluster


def kmeans(X, k):
    """ performs K-means on a dataset
    Arg:
        - X: np.ndarray of shape (n, d) containing the dataset
            - n is the number of data points
            - d is the number of dimensions for each data point
        - k: positive int containing the number of clusters

    Returns: (C, clss) or (None, None) on failure
        - C: np.ndarray (k, d) with the centroid means for each cluster
        - clss: np.ndarray (n,) with the index of the cluster in C that
                each data point belongs to
    """
    Kmean = sklearn.cluster.KMeans(n_clusters=k)
    Kmean.fit(X)

    C = Kmean.cluster_centers_
    clss = Kmean.labels_

    return C, clss

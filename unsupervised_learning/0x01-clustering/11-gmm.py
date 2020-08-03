#!/usr/bin/env python3
"""
Kmean
"""

import sklearn.mixture


def gmm(X, k):
    """ calculates a GMM from a dataset
    Arg:
        - X: np.ndarray of shape (n, d) containing the dataset
            - n is the number of data points
            - d is the number of dimensions for each data point
        - k: positive int containing the number of clusters

    Returns: pi, m, S, clss, bic

        - pi is a numpy.ndarray of shape (k,) containing the cluster priors.
        - m is a numpy.ndarray of shape (k, d) containing the centroid means.
        - S is a numpy.ndarray of shape (k, d, d) containing the covariance
          matrices.
        - clss is a numpy.ndarray of shape (n,) containing the cluster indices
          for each data point
        - bic is a numpy.ndarray of shape (kmax - kmin + 1) containing the BIC
        value for each cluster size tested.
    """
    g = sklearn.mixture.GaussianMixture(n_components=k)
    g.fit(X)

    pi = g.weights_
    m = g.means_
    S = g.covariances_
    clss = g.predict(X)
    bic = g.bic(X)

    return pi, m, S, clss, bic

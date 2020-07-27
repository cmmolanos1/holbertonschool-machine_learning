#!/usr/bin/env python3
"""
Mean and covariance
"""
import numpy as np


def mean_cov(X):
    """Calculates the mean and covariance of a data set.

    Args:
        X (np.ndarray): dataset of shape (n, d)

    Returns:
        mean and covariance matrix of dataset.
    """
    if type(X) is not np.ndarray or len(X.shape) != 2:
        raise TypeError("X must be a 2D numpy.ndarray")

    if X.shape[0] < 2:
        raise ValueError("X must contain multiple data points")

    n, d = X.shape
    mean = np.mean(X, axis=0).reshape((1, d))

    ones = np.ones((n, n))
    std_scores = X - np.matmul(ones, X) * (1 / n)

    cov_matrix = np.matmul(std_scores.T, std_scores) / (n - 1)

    return mean, cov_matrix

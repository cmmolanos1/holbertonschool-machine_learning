#!/usr/bin/env python3
"""
Multinormal
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

    d, n = X.shape
    mean = np.mean(X, axis=1).reshape((d, 1))

    ones = np.ones((n, n))
    std_scores = X - np.matmul(X, ones) * (1 / n)

    cov_matrix = np.matmul(std_scores, std_scores.T) / (n - 1)

    return mean, cov_matrix


class MultiNormal():
    """
        Class multinormal
    """

    def __init__(self, data):
        """
        Class constructor
        """
        if type(data) is not np.ndarray:
            raise TypeError("data must be a 2D numpy.ndarray")

        d, n = data.shape

        if n < 2:
            raise ValueError("data must contain multiple data points")

        self.mean, self.cov = mean_cov(data)

    def pdf(self, x):
        """Calculates the Probability distribution function at a data point.

        Args:
            x (np.ndarray): matrix of shape (d, 1) containing the data point
                            whose PDF should be calculated.

        Returns:

        """
        if type(x) is not np.ndarray:
            raise TypeError("x must by a numpy.ndarray")

        d, _ = x.shape
        if x.shape != (d, 1):
            raise ValueError("x must have the shape ({}, 1)".format(d))

        x_minusu = x - self.mean
        cov_det = np.linalg.det(self.cov)
        cov_inv = np.linalg.inv(self.cov)

        pdf1 = 1 / np.sqrt(((2 * np.pi) ** d) * cov_det)
        pdf2 = np.exp(-0.5 * np.matmul(np.matmul(x_minusu.T, cov_inv)),
                      x_minusu)

        pdf = pdf1 * pdf2

        return pdf

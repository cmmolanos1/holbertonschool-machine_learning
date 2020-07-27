#!/usr/bin/env python3
"""
Multinormal
"""

import numpy as np


class MultiNormal():
    """
        Class multinormal
    """

    def __init__(self, data):
        """
        Class constructor
        """
        if type(data) is not np.ndarray or len(data.shape) != 2:
            raise TypeError("data must be a 2D numpy.ndarray")
        d, n = data.shape
        if n < 2:
            raise ValueError("data must contain multiple data points")

        self.mean = (np.mean(data, axis=1)).reshape(d, 1)

        X = data - self.mean
        self.cov = ((np.matmul(X, X.T)) / (n - 1))

    def pdf(self, x):
        """Calculates the Probability distribution function at a data point.

        Args:
            x (np.ndarray): matrix of shape (d, 1) containing the data point
                            whose PDF should be calculated.

        Returns:

        """
        if not isinstance(x, np.ndarray):
            raise TypeError("x must be a numpy.ndarray")
        d = self.cov.shape[0]
        if len(x.shape) != 2 or x.shape[1] != 1 or x.shape[0] != d:
            raise ValueError("x must have the shape ({}, 1)".format(d))

        # pdf formula -- multivar

        det = np.linalg.det(self.cov)
        inv = np.linalg.inv(self.cov)
        f1 = 1 / np.sqrt(((2 * np.pi) ** d) * det)
        f21 = -(x - self.mean).T
        f22 = np.matmul(f21, inv)
        f23 = (x - self.mean) / 2
        f24 = np.matmul(f22, f23)
        f2 = np.exp(f24)
        pdf = f1 * f2

        return pdf.flatten()[0]

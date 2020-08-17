#!/usr/bin/env python3
"""
Gaussian
"""

import numpy as np


class GaussianProcess():
    """
    Gaussian Class
    """

    def __init__(self, X_init, Y_init, l=1, sigma_f=1):
        """Class constructor

        Args:
            X_init (np.ndarray): shape (t, 1) representing the inputs already
                                 sampled with the black-box function.
            Y_init (np.ndarray): shape (t, 1) representing the outputs of the
                                 black-box function for each input in X_init.
            l (int): the length parameter for the kernel.
            sigma_f (int): the standard deviation given to the output of the
                           black-box function.
        """
        self.X = X_init
        self.Y = Y_init
        self.l = l
        self.sigma_f = sigma_f
        self.K = self.kernel(X_init, X_init)

    def kernel(self, X1, X2):
        """calculates the covariance kernel matrix between two matrices

        Args:
            X1 (np.ndarray): shape (m, 1)
            X2 (np.ndarray): shape (n, 1)

        Returns:
             the covariance kernel matrix as a numpy.ndarray of shape (m, n).
        """
        a = np.sum(X1 ** 2, 1).reshape(-1, 1)
        sqdist = a + np.sum(X2 ** 2, 1) - 2 * np.dot(X1, X2.T)

        return self.sigma_f ** 2 * np.exp(-0.5 / self.l ** 2 * sqdist)

    def predict(self, X_s):
        """predicts the mean and standard deviation of points in a Gaussian
        process.

        Args:
            X_s (np.ndarray): shape (s, 1) containing all of the points whose
            mean and standard deviation should be calculated. S: number of
            sample points.

        Returns:
            Returns: mu, sigma
            - mu is a numpy.ndarray of shape (s,) containing the mean for each
            point in X_s, respectively.
            - sigma is a numpy.ndarray of shape (s,) containing the standard
            deviation for each point in X_s, respectively.
        """
        K_s = self.kernel(self.X, X_s)
        K_ss = self.kernel(X_s, X_s) + 1e-8 * np.eye(len(X_s))
        K_inv = np.linalg.inv(self.K)

        # μ∗ = K∗.T * inv(Ky) * y
        mu_s = K_s.T.dot(K_inv).dot(self.Y)

        # Σ∗ = K∗∗ − K∗.T * inv(Ky) * K∗
        cov_s = K_ss - K_s.T.dot(K_inv).dot(K_s)

        mu = np.reshape(mu_s, -1)
        sigma = np.diagonal(cov_s)

        return mu, sigma

    def update(self, X_new, Y_new):
        """Updates a gaussian process

        Args:
            X_new (np.ndarray): shape (1,) that represents the new sample
                                point.
            Y_new (np.ndarray): shape (1,) that represents the new sample
                                function value.
        """
        self.X = np.vstack((self.X, X_new))
        self.Y = np.vstack((self.Y, Y_new))
        self.K = self.kernel(self.X, self.X)

#!/usr/bin/env python3
"""
Bayesian
Based from: http://krasserm.github.io/2018/03/21/bayesian-optimization/
"""

import numpy as np
from scipy.stats import norm

GP = __import__('2-gp').GaussianProcess


class BayesianOptimization():
    """
    Bayesian
    """

    def __init__(self, f, X_init, Y_init, bounds, ac_samples, l=1, sigma_f=1,
                 xsi=0.01, minimize=True):
        """ Class constructor

        Args:
            f: the black-box function  to be optimized.
            X_init (np.ndarray): shape (t, 1) representing the inputs already
                                 sampled with the black-box function.
            Y_init (np.ndarray): shape (t, 1) representing the outputs of the
                                 black-box function for each input in X_init.
                                 t: number of initial samples.
            bounds (tuple): (min, max) representing the bounds of the space in
                             which to look for the optimal point.
            ac_samples (int): the number of samples that should be analyzed
                              during acquisition.
            l (int): the length parameter for the kernel.
            sigma_f (float): the standard deviation given to the output of the
                             black-box function.
            xsi: the exploration-exploitation factor for acquisition
            minimize (bool): determining whether optimization should be
                             performed for minimization (True) or maximization
                             (False).
        """
        self.f = f
        self.gp = GP(X_init, Y_init, l, sigma_f)
        self.X_s = np.linspace(bounds[0],
                               bounds[1],
                               ac_samples).reshape(ac_samples, 1)
        self.xsi = xsi
        self.minimize = minimize

    def acquisition(self):
        """calculates the next best sample location.

        Returns:
            X_next, EI

            - X_next is a numpy.ndarray of shape (1,) representing the next
              best sample point.
            - EI is a numpy.ndarray of shape (ac_samples,) containing the
            expected improvement of each potential sample.
        """
        mu_sample, sigma_sample = self.gp.predict(self.X_s)

        if self.minimize is True:
            Y_sample_opt = np.min(self.gp.Y)
            imp = Y_sample_opt - mu_sample - self.xsi
        else:
            Y_sample_opt = np.max(self.gp.Y)
            imp = mu_sample - Y_sample_opt - self.xsi

        with np.errstate(divide='ignore'):
            Z = imp / sigma_sample
            ei = imp * norm.cdf(Z) + sigma_sample * norm.pdf(Z)
            ei[sigma_sample == 0.0] = 0.0

        X_next = self.X_s[np.argmax(ei)]

        return (X_next, ei)

    def optimize(self, iterations=100):
        """optimizes the black-box function.

        Args:
            iterations(int): the maximum number of iterations to perform.

        Returns:
            Returns: X_opt, Y_opt
            - X_opt is a numpy.ndarray of shape (1,) representing the optimal
              point.
            - Y_opt is a numpy.ndarray of shape (1,) representing the optimal
              function value
        """
        for i in range(iterations):
            # Obtain next sampling point from the acquisition function
            # (expected_improvement)
            X_next, _ = self.acquisition()
            # Check if X_next was sampled.
            if X_next in self.gp.X:
                break

            # Obtain next noisy sample from the objective function
            Y_next = self.f(X_next)

            # Add sample to previous samples
            self.gp.update(X_next, Y_next)

        if self.minimize is True:
            X_opt = np.amin(self.gp.Y)

        else:
            X_opt = np.amax(self.gp.Y)

        Y_opt = self.f(X_opt)

        return X_opt, Y_opt

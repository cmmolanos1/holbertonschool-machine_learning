#!/usr/bin/env python3
"""Class Poisson"""

# Euler
e = 2.7182818285


class Exponential():
    """ Class to calculate Exponential distribution"""

    def __init__(self, data=None, lambtha=1.):
        """Constructor of Poisson

        Args:
            data (list): dataset of distribution
            lambtha (float): n*p
        """
        self.lambtha = float(lambtha)
        if data is None:
            if lambtha <= 0:
                raise ValueError("lambtha must be a positive value")
        else:
            if isinstance(data, list):
                if len(data) > 1:
                    self.data = data
                    self.lambtha = 1 / (sum(self.data) / len(self.data))
                else:
                    raise ValueError("data must contain multiple values")
            else:
                raise TypeError("data must be a list")

    def pdf(self, k):
        """ Calculates the Probability Density Function of the distribution.
        f(x;λ) = λe^(-λx)

        Args:
            k (int): number of successes.

        Returns:
            float: the value of the distribution function at point k.
        """
        x = k
        if x < 0:
            return 0
        else:
            f = self.lambtha * e ** (-self.lambtha * x)
            return f

    def cdf(self, k):
        """ Calculates the cumulative distribution function.
        f(x;λ) = -e^(-λx) + c from 0 to x

        Args:
            k: number of successes.

        Returns:
            float: the integral of the function from 0 to k.
        """
        x = k
        if x < 0:
            return 0
        cdf_exp = -e ** (-self.lambtha * x) + 1
        return cdf_exp

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
            if lambtha > 0:
                self.data = lambtha
            else:
                raise ValueError("lambtha must be a positive value")
        else:
            if not isinstance(data, list):
                raise TypeError("data must be a list")
            elif len(data) < 2:
                raise ValueError("data must contain multiple values")
            else:
                self.data = data
                self.lambtha = 1 / (sum(self.data) / len(self.data))

    def pdf(self, k):
        """ Calculates the Probability Density Function of the distribution.
        f(x;λ) = λe^(-λx)

        Args:
            k (int): number of successes.

        Returns:
            float: the value of the distribution function at point k.
        """
        x = k
        if (isinstance(self.data, float) and x > 1) or \
                (isinstance(self.data, list) and x > len(self.data)):
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
        if (isinstance(self.data, float) and x > 1) or \
                (isinstance(self.data, list) and x > len(self.data)):
            return 0
        cdf_exp = -e ** (-self.lambtha * x) + 1
        return cdf_exp

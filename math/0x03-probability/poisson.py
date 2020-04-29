#!/usr/bin/env python3
"""Class Poisson"""

# Euler
e = 2.7182818285


class Poisson():
    """ Class to calculate Poisson distribution"""

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
                    self.lambtha = float(sum(self.data) / len(self.data))
                else:
                    raise ValueError("data must contain multiple values")
            else:
                raise TypeError("data must be a list")

    def pmf(self, k):
        """ Calculates the Probability Mass Function of the distribution.

        Args:
            k (int): number of successes.

        Returns:
            float: the value of the distribution function of k.
        """
        x = int(k)
        if x < 0:
            return 0
        else:
            x_fact = 1
            for i in range(1, x + 1):
                x_fact = x_fact * i

            f = e ** -self.lambtha * self.lambtha ** x / x_fact
            return f

    def cdf(self, k):
        """ Calculates the cumulative distribution function

        Args:
            k: number of successes.

        Returns:
            float: the sum of all pmf from 0 to k.
        """
        k = int(k)
        if k < 0:
            return 0

        cdf = 0
        for i in range(k + 1):
            cdf += self.pmf(i)
        return cdf

#!/usr/bin/env python3
"""Class Normal"""

e = 2.7182818285
pi = 3.1415926536


class Normal():
    """ Class to calculate Normal distribution"""

    def __init__(self, data=None, mean=0., stddev=1.):
        """Initialize the distribution

        Args:
            data (list): values of distribution.
            mean (float): mean of distribution.
            stddev (float): standard deviation of distribution.
        """
        self.mean = float(mean)
        self.stddev = float(stddev)

        if data is None:
            if self.stddev < 0:
                raise ValueError("stddev must be a positive value")
        else:
            if isinstance(data, list):
                if len(data) > 1:
                    self.mean = sum(data) / len(data)
                    n = len(data)
                    variance = sum([(n - self.mean) ** 2 for n in data]) / n
                    self.stddev = variance ** 0.5
                else:
                    raise ValueError("data must contain multiple values")

            else:
                raise TypeError("data must be a list")

    def z_score(self, x):
        """Calculates z

        Args:
            x (float): x-value.

        Returns:
            float: z value
        """
        return (x - self.mean) / self.stddev

    def x_value(self, z):
        """Calculates x from z

        Args:
            z (float): z-score

        Returns:
            float: x-score
        """
        return z * self.stddev + self.mean

    def pdf(self, x):
        """Calculates the function

        Args:
            x (float): x-value.

        Returns:
            float: the probability at x-value.
        """
        constant = 1 / (self.stddev * (2 * pi) ** 0.5)
        exponent = e ** -(((x - self.mean) ** 2) / (2 * self.stddev ** 2))
        return constant * exponent

    def cdf(self, x):
        """ Calculates the cummulative distribution function.

        Args:
            x (float): x-value.

        Returns:
            float: the CDF.
        """
        b = (x - self.mean) / (self.stddev * (2 ** 0.5))
        erf = (2 / (pi ** 0.5)) * \
              (b - (b ** 3) / 3 + (b ** 5) / 10 - (b ** 7) / 42 +
               (b ** 9) / 216)
        return 0.5 * (1 + erf)

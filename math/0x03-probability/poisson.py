#!/usr/bin/env python3
"""Class Poisson"""

e = 2.7182818285


class Poisson():
    def __init__(self, data=None, lambtha=1.):
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
                self.lambtha = sum(self.data) / len(self.data)

    def pmf(self, k):
        x = int(k)
        if (isinstance(self.data, float) and x > 1) or \
                (isinstance(self.data, list) and x > len(self.data)):
            return 0
        else:
            x_fact = 1
            for i in range(1, x + 1):
                x_fact = x_fact * i

            f = e ** -self.lambtha * self.lambtha ** x / x_fact
            return f

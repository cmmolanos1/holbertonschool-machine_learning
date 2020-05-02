#!/usr/bin/env python3
"""First Neuron"""
import numpy as np


class Neuron():
    """Defines a single neuron performing binary classification"""

    def __init__(self, nx):
        """Constructor.

        Args:
            nx (int): number of inputs (Xi).
        """
        if type(nx) is not int:
            raise TypeError("nx must be an integer")
        if nx < 1:
            raise ValueError("nx must be a positive integer")

        self.nx = nx
        # Private Weight
        self.__W = np.random.randn(nx).reshape(1, nx)
        # Private Bias
        self.__b = 0
        # Private Output
        self.__A = 0

    @property
    def W(self):
        """ W attribute getter.

        Returns:
            ndarray: Private W.
        """
        return self.__W

    @property
    def b(self):
        """ b attribute getter.

        Returns:
            int: Private b.
        """
        return self.__b

    @property
    def A(self):
        """ A attribute getter.

        Returns:
            int: Private A.
        """
        return self.__A

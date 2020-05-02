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
        # Weight
        self.W = np.random.randn(nx).reshape(1, nx)
        # Bias
        self.b = 0
        # Output
        self.A = 0

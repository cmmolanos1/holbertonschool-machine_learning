#!/usr/bin/env python3
"""First Neural Network"""
import numpy as np


class NeuralNetwork():
    """Defines a neural network performing binary classification"""

    def __init__(self, nx, nodes):
        """ Class constructor.

        Args:
            nx (int): number of input features.
            nodes (int): number of nodes found in the hidden layer.
        """
        if type(nx) is not int:
            raise TypeError("nx must be an integer")
        if nx < 1:
            raise ValueError("nx must be a positive integer")
        if type(nodes) is not int:
            raise TypeError("nodes must be an integer")
        if nodes < 1:
            raise TypeError("nodes must be a positive integer")

        # Hidden layer.
        self.__W1 = np.random.randn(nodes * nx).reshape(nodes, nx)
        self.__b1 = np.zeros((nodes, 1))
        self.__A1 = 0
        # Output neuron.
        self.__W2 = np.random.randn(1, nodes)
        self.__b2 = 0
        self.__A2 = 0

    @property
    def W1(self):
        """ W1 attribute getter.

        Returns:
            ndarray: Private W1.
        """
        return self.__W1

    @property
    def b1(self):
        """ b1 attribute getter.

        Returns:
            int: Private b1.
        """
        return self.__b1

    @property
    def A1(self):
        """ A1 attribute getter.

        Returns:
            int: Private A1.
        """
        return self.__A1

    @property
    def W2(self):
        """ W2 attribute getter.

        Returns:
            ndarray: Private W2.
        """
        return self.__W2

    @property
    def b2(self):
        """ b2 attribute getter.

        Returns:
            int: Private b2.
        """
        return self.__b2

    @property
    def A2(self):
        """ A2 attribute getter.

        Returns:
            int: Private A2.
        """
        return self.__A2

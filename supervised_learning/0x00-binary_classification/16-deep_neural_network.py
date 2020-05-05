#!/usr/bin/env python3
"""First Deep Neural Network"""
import numpy as np


class DeepNeuralNetwork():
    """Defines a deep neural network performing binary classification"""

    def __init__(self, nx, layers):
        """Constructor.

        Args:
            nx (int): number of inputs (Xi).
            layers (list): the number of nodes in each layer of the network.
        """
        if type(nx) is not int:
            raise TypeError("nx must be an integer")
        if nx < 1:
            raise ValueError("nx must be a positive integer")
        if type(layers) is not list:
            raise TypeError("layers must be a list of positive integers")
        arr_layers = np.array(layers)
        len_pos = arr_layers[arr_layers >= 1].shape[0]
        if len(layers) == 0 or len_pos != len(layers) or \
                isinstance(arr_layers[0], np.integer) is False:
            raise TypeError("layers must be a list of positive integers")

        self.L = len(layers)
        self.cache = {}
        self.weights = {}

        for la in range(self.L):
            # Layer 0 = X, Layers = [L1_size, L2_size, ..., Ln_size]
            key_w = "W{}".format(la + 1)
            key_b = "b{}".format(la + 1)

            if la == 0:
                # Layer 0 = X, Size Layer 0 = nx
                weight = np.random.randn(layers[la], nx) * np.sqrt(2 / nx)
                self.weights[key_w] = weight
            else:
                weight = np.random.randn(layers[la], layers[la - 1]) * \
                         np.sqrt(2 / layers[la - 1])
                self.weights[key_w] = weight

            biases = np.zeros(layers[la]).reshape(layers[la], 1)
            self.weights[key_b] = biases

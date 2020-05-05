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

        self.__L = len(layers)
        self.__cache = {}
        self.__weights = {}

        for la in range(self.__L):
            # Layer 0 = X, Layers = [L1_size, L2_size, ..., Ln_size]
            key_w = "W{}".format(la + 1)
            key_b = "b{}".format(la + 1)

            if la == 0:
                # Layer 0 = X, Size Layer 0 = nx
                weight = np.random.randn(layers[la], nx) * np.sqrt(2 / nx)
                self.__weights[key_w] = weight
            else:
                weight = np.random.randn(layers[la], layers[la - 1]) * \
                         np.sqrt(2 / layers[la - 1])
                self.__weights[key_w] = weight

            biases = np.zeros((layers[la], 1))
            self.__weights[key_b] = biases

    @property
    def L(self):
        """ A attribute getter.

        Returns:
            int: Private A.
        """
        return self.__L

    @property
    def cache(self):
        """ cache attribute getter.

        Returns:
            int: Private A.
        """
        return self.__cache

    @property
    def weights(self):
        """ A attribute getter.

        Returns:
            int: Private A.
        """
        return self.__weights

    def forward_prop(self, X):
        """ Calculates forward propagation of the neural network.

        Args:
            X (ndarray): input data (number X, number examples).

        Returns:
            ndarray: vector with activation results using sigmoid.
            dict: dictionary with all activation vectors.
        """
        self.__cache['A0'] = X

        for la in range(self.__L):
            key_W = "W{}".format(la + 1)
            key_b = "b{}".format(la + 1)
            key_A = "A{}".format(la)
            key_newA = "A{}".format(la + 1)

            W = self.__weights[key_W]
            A = self.__cache[key_A]
            b = self.__weights[key_b]
            z = np.matmul(W, A) + b
            sigmoid = 1 / (1 + np.exp(-z))
            self.__cache[key_newA] = sigmoid

        return sigmoid, self.__cache

    def cost(self, Y, A):
        """Calculates the function cost of the neuron.
        It was applied the Logistic Regression Cost Function.

        Args:
            Y (ndarray): the actual or correct values for the input data.
            A (ndarray): activated output of the neuron for each example.

        Returns:
            float: calculated cost, better if cost -> 0 .
        """
        constant = -1 / A.shape[1]
        summation = Y * np.log(A) + (1 - Y) * np.log(1.0000001 - A)
        return constant * summation.sum()

    def evaluate(self, X, Y):
        """Evaluates neural network predictions.

        Args:
            X (ndarray): input data shape(nx, m).
            Y (ndarray): the actual or correct values for the input data.

        Returns:
            ndarray: activated output of neuron, where if A[i] >= 0.5
                     the output is 1, otherwise 0.
            float: the neuron's cost.
        """
        A, _ = self.forward_prop(X)
        cost = self.cost(Y, A)
        A = np.where(A >= 0.5, 1, 0)
        return A, cost

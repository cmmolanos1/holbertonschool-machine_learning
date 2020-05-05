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

    def forward_prop(self, X):
        """ Calculates neuron output.

        Args:
            X (ndarray): rows X's, cols examples.

        Returns:
            ndarray: vector with activation results using sigmoid.
        """
        # sum(W.X)
        sum_WxX = np.matmul(self.__W, X)
        # -sum(W.X) - b
        z = sum_WxX + self.__b
        # Sigmoid = 1 / 1 + e^(-sum(W.X) - b)
        sigmoid_z = 1 / (1 + np.exp(-z))
        self.__A = sigmoid_z
        return self.__A

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
        """Evaluates neuron predictions.

        Args:
            X (ndarray): input data shape(nx, m).
            Y (ndarray): the actual or correct values for the input data.

        Returns:
            ndarray: activated output of neuron, where if A[i] >= 0.5
                     the output is 1, otherwise 0.
            float: the neuron's cost.
        """
        self.forward_prop(X)
        cost = self.cost(Y, self.__A)
        return np.where(self.__A >= 0.5, 1, 0), cost

    def gradient_descent(self, X, Y, A, alpha=0.05):
        """Calculates one pass of gradient descent on the neuron.

        Args:
            X (ndarray): input data shape(nx, m).
            Y (ndarray): the actual or correct values for the input data.
            A (ndarray): activated output of the neuron for each example.
            alpha (float): learning rate.

        Returns:
            ndarray: array with corrected weights.
            float: corrected bias.
        """
        m = Y.shape[1]
        dz = A - Y
        dW = np.dot(X, dz.T).T / m
        db = dz.sum() / m
        self.__W -= alpha * dW
        self.__b -= alpha * db

    def train(self, X, Y, iterations=5000, alpha=0.05):
        """Trains the neuron, performing gradient_descent iterations times.

        Args:
            X (ndarray): input data shape(nx, m).
            Y (ndarray): the actual or correct values for the input data.
            iterations (int): number of times the function will train the
                              neuron.
            alpha (float): learning rate.

        Returns:
            ndarray: activated output of neuron at the end of training,
                     where if A[i] >= 0.5 the output is 1, otherwise 0.
            float: the neuron's cost at the end of training.
        """
        if type(iterations) is not int:
            raise TypeError("iterations must be an integer")
        if iterations <= 0:
            raise ValueError("iterations must be a positive integer")
        if type(alpha) is not float:
            raise TypeError("alpha must be a float")
        if alpha <= 0:
            raise ValueError("alpha must be positive")

        for i in range(iterations):
            self.forward_prop(X)
            self.gradient_descent(X, Y, self.__A, alpha)

        return self.evaluate(X, Y)

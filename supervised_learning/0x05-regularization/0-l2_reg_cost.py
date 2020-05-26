#!/usr/bin/env python3
"""
L2 Regularization
"""

import numpy as np


def l2_reg_cost(cost, lambtha, weights, L, m):
    """Calculates the cost of a neural network with L2 regularization.

    Args:
        cost (float): cost of the network without L2 regularization.
        lambtha (float): regularization parameter.
        weights (dict): the weights and biases (numpy.ndarrays) of the
                        neural network.
        L (int): number of layers in the neural network.
        m (int): number of data points used.

    Returns:
        np.ndarray: the cost of the network accounting for L2 regularization.
    """
    summation = 0
    for ly in range(1, L + 1):
        key = "W{}".format(ly)
        summation += np.linalg.norm(weights[key])

    L2_cost = lambtha * summation / (2 * m)

    return cost + L2_cost

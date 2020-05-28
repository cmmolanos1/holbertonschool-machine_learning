#!/usr/bin/env python3
"""
L2 Regularization
"""

import numpy as np


def l2_reg_gradient_descent(Y, weights, cache, alpha, lambtha, L):
    """Updates the weights and biases of a neural network using gradient
       descent with L2 regularization.

    Args:
        Y (np.ndarray): one-hot matrix of shape (classes, m) that contains
                        the correct labels for the data.
        weights (dict): the weights and biases of the neural network.
        cache (dict): the outputs of each layer of the neural network.
        alpha (float):  learning rate.
        lambtha (float): L2 regularization parameter.
        L (int): number of layers of the network.

    Returns:
        dict: weights and biases of the network should be updated in place.
    """
    m = Y.shape[1]
    dz = {}
    dW = {}
    db = {}
    for la in reversed(range(1, L + 1)):
        A = cache["A{}".format(la)]
        A_prev = cache["A{}".format(la - 1)]

        # 3
        if la == L:
            kdz = "dz{}".format(la)
            kdW = "dW{}".format(la)
            kW = "W{}".format(la)
            kdb = "db{}".format(la)

            W = weights[kW]
            dz[kdz] = A - Y
            dW[kdW] = np.matmul(dz[kdz], A_prev.T) / m + (lambtha * W) / m
            db[kdb] = dz[kdz].sum(axis=1, keepdims=True) / m
        else:
            # 2 - 1
            kdz_n = "dz{}".format(la + 1)
            kdz_c = "dz{}".format(la)
            kdW_n = "dW{}".format(la + 1)
            kdW = "dW{}".format(la)
            kdb_n = "db{}".format(la + 1)
            kdb = "db{}".format(la)
            kW = 'W{}'.format(la + 1)
            kW_prev = 'W{}'.format(la)
            kb = 'b{}'.format(la + 1)

            W = weights[kW]
            W_p = weights[kW_prev]
            g = 1 - (A * A)
            dz[kdz_c] = np.matmul(W.T, dz[kdz_n]) * g
            dW[kdW] = np.matmul(dz[kdz_c], A_prev.T) / m + (lambtha * W_p) / m
            db[kdb] = dz[kdz_c].sum(axis=1, keepdims=True) / m

            weights[kW] -= alpha * dW[kdW_n]
            weights[kb] -= alpha * db[kdb_n]

            if la == 1:
                weights['W1'] -= alpha * dW['dW1']
                weights['b1'] -= alpha * db['db1']

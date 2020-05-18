#!/usr/bin/env python3
"""
RMS
"""

import numpy as np


def update_variables_Adam(alpha, beta1, beta2, epsilon, var, grad, v, s, t):
    """Updates a variable in place using the Adam optimization algorithm.

    Args:
        alpha (float): learning rate.
        beta1 (float): weight used for the first moment.
        beta2 (float): weight used for the second moment.
        epsilon (float): small number to avoid division by zero.
        var (np.ndarray): the variable to be updated.
        grad (np.ndarray): the gradient of var.
        v (np.ndarray): previous first moment of var.
        s (np.ndarray): previous second moment of var.
        t (int): time step used for bias correction.

    Returns:
        the updated variable, the new first moment, and the new second moment.
    """
    v = beta1 * v + (1 - beta1) * grad
    s = beta2 * s + (1 - beta2) * (grad * grad)

    v_corrected = v / (1 - beta1 ** t)
    s_corrected = s / (1 - beta2 ** t)

    var -= alpha * v_corrected / (np.sqrt(s_corrected) + epsilon)

    return var, v, s

#!/usr/bin/env python3
"""
Momentum
"""

import numpy as np


def update_variables_RMSProp(alpha, beta2, epsilon, var, grad, s):
    """Updates a variable using the RMSProp optimization algorithm.

    Args:
        alpha (float): learning rate.
        beta2 (float): RMSProp weight.
        epsilon (float): small number to avoid division by zero.
        var (np.ndarray): the variable to be updated.
        grad (np.ndarray): the gradient of var.
        s (np.ndarray): the previous second moment of var

    Returns:
        np.ndarray:  the updated variable and the new moment, respectively.
    """
    s = beta2 * s + (1 - beta2) * (grad * grad)
    var -= alpha * grad / (np.sqrt(s) + epsilon)
    return var, s

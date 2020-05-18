#!/usr/bin/env python3
"""
Momentum
"""


def update_variables_momentum(alpha, beta1, var, grad, v):
    """Updates a variable using the gradient descent with momentum
    optimization algorithm.

    Args:
        alpha (float): learning late.
        beta1 (float): momentum weight.
        var (np.ndarray): variable to be updated.
        grad (np.ndarray): gradient of var.
        v (np.ndarray): the previous first moment of var.

    Returns:
        np.ndarray: the updated variable and the new moment.
    """
    v = beta1 * v + (1 - beta1) * grad
    var -= alpha * v
    return var, v

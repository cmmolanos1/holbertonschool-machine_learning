#!/usr/bin/env python3
"""
Bias correction
"""


def moving_average(data, beta):
    """Calculates the weighted moving average of a data set.

    Args:
        data (list): data to calculate the moving average of.
        beta (float): weight used for the moving average.

    Returns:
        list: moving averages of data.
    """
    Vt = 0
    W_averages = []
    for i in range(len(data)):
        Vt = beta * Vt + (1 - beta) * data[i]
        b_correction = 1 - beta ** (i + 1)
        W_averages.append(Vt / b_correction)

    return W_averages

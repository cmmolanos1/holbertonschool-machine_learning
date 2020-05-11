#!/usr/bin/env python3

import numpy as np


def one_hot_decode(one_hot):
    """Converts a one-hot matrix into a vector of labels.

    Args:
        one_hot (numpy.ndarray):  one how matrix with shape (classes, m).

    Returns:
        numpy.ndarray: labels vector.
    """
    if isinstance(one_hot, np.ndarray) is False:
        return 0

    return np.argmax(one_hot, axis=0)

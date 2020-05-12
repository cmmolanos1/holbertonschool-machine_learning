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

# def one_hot_decode(one_hot):
#     """ converts a one-hot matrix into a vector of labels """
#     if not isinstance(one_hot, np.ndarray) or len(one_hot.shape) != 2:
#         return None
#     if not np.where((one_hot == 0) | (one_hot == 1), True, False).all():
#         return None
#     if np.sum(one_hot) != len(one_hot[0]):
#         return None
#     classes = len(one_hot)
#     m = len(one_hot[0])
#     samples = np.zeros(m)
#     tmp = np.arange(m)
#     axis = np.argmax(one_hot, axis=0)
#     samples[tmp] = axis
#     return samples.astype("int64")
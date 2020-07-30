#!/usr/bin/env python3
"""
PSA
"""

import numpy as np


def pca(X, var=0.95):
    """Performs PCA on a dataset
    """
    u, s, vh = np.linalg.svd(X)
    total_variance = np.cumsum(s) / np.sum(s)
    r = (np.argwhere(total_variance >= var))[0, 0]

    weight = vh[:r + 1].T

    return weight

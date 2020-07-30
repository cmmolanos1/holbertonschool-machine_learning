#!/usr/bin/env python3
"""
PCA
"""

import numpy as np


def pca(X, ndim):
    """
    Performs PCA on a dataset
    """
    X_m = X - np.mean(X, axis=0)

    u, s, vh = np.linalg.svd(X_m)
    W = vh[:ndim].T

    T = np.matmul(X_m, (W))

    return T

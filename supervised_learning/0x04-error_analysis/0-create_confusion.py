#!/usr/bin/env python3
"""
Confusion Matrix
"""

import numpy as np


def create_confusion_matrix(labels, logits):
    """Creates a confusion matrix.

    Args:
        labels (np.ndarray): one-hot of shape (m, classes) containing the
                             correct labels for each data point.
        logits (np.ndarray): one-hot of shape (m, classes) containing the
                             predicted labels for each data point.

    Returns:
        np.ndarray: a confusion matrix of shape (classes, classes) with row
                    indices representing the correct labels and column indices
                    representing the predicted labels.
    """
    return np.matmul(labels.T, logits)

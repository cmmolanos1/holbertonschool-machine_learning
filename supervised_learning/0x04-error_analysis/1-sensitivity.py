#!/usr/bin/env python3

import numpy as np


def sensitivity(confusion):
    """Calculates the sensitivity for each class in a confusion matrix.

    Args:
        confusion (np.ndarray): confusion matrix of shape (classes, classes)
                                where row indices represent the correct
                                labels and column indices represent the
                                predicted labels.

    Returns:
        np.ndarray: array of shape (classes,) containing the sensitivity of
                    each class.
    """
    true_positive = confusion.diagonal()
    # TP + FN
    positives = np.sum(confusion, axis=1).T

    return true_positive / positives

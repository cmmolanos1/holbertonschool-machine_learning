#!/usr/bin/env python3

import numpy as np


def precision(confusion):
    """Calculates the precision for each class in a confusion matrix.

        Args:
            confusion (np.ndarray): confusion matrix of shape (classes, classes)
                                    where row indices represent the correct
                                    labels and column indices represent the
                                    predicted labels.

        Returns:
            np.ndarray: array of shape (classes,) containing the precision of
                        each class.
    """
    true_positive = confusion.diagonal()
    # TP + FP
    TP_FP = np.sum(confusion, axis=0)

    return true_positive / TP_FP

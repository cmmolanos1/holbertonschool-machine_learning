#!/usr/bin/env python3

import numpy as np


def specificity(confusion):
    """Calculates the specificity for each class in a confusion matrix.

        Args:
            confusion (np.ndarray): confusion matrix of shape(classes,classes)
                                    where row indices represent the correct
                                    labels and column indices represent the
                                    predicted labels.

        Returns:
            np.ndarray: array of shape (classes,) containing the specificity of
                        each class.
    """
    classes = confusion.shape[0]

    total = np.array([np.sum(confusion)] * classes)
    FN = np.sum(confusion, axis=0)
    FP = np.sum(confusion, axis=1)
    TP = confusion.diagonal()

    TN = total - FN - FP + TP
    FP = FN - TP

    return TN / (TN + FP)

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
    TNR = np.zeros((classes,))
    for i in range(classes):
        TN = np.sum(confusion) - np.sum(confusion[i])\
             - np.sum(confusion[:, i]) + confusion[i, i]
        FP = np.sum(confusion[:, i]) - confusion[i, i]
        TNR[i] = TN / (TN + FP)

    return TNR

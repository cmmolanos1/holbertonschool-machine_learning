#!/usr/bin/env python3
"""
F1
"""

sensitivity = __import__('1-sensitivity').sensitivity
precision = __import__('2-precision').precision


def f1_score(confusion):
    """Calculates the F1-score for each class in a confusion matrix.

        Args:
            confusion (np.ndarray): confusion matrix of shape(classes,classes)
                                    where row indices represent the correct
                                    labels and column indices represent the
                                    predicted labels.

        Returns:
            np.ndarray: array of shape (classes,) containing the F1-score of
                        each class.
    """
    PPV = precision(confusion)
    TPR = sensitivity(confusion)
    return 2 * (PPV * TPR) / (PPV + TPR)

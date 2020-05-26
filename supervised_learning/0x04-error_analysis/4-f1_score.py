#!/usr/bin/env python3
"""
F1 Score
"""

sensitivity = __import__('1-sensitivity').sensitivity
precision = __import__('2-precision').precision


def f1_score(confusion):
    """Calculates the F1-score for each class in a confusion matrix.

        Args:
            confusion: confusion matrix of shape(classes,classes)
                                    where row indices represent the correct
                                    labels and column indices represent the
                                    predicted labels.

        Returns:
            np.array: array of shape (classes,) containing the F1-score of
                        each class.
    """
    PPV = precision(confusion)
    TPR = sensitivity(confusion)
    F1 = 2 * PPV * TPR / (PPV + TPR)
    return F1

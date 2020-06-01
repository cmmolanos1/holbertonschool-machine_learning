#!/usr/bin/env python3
"""
Models
"""

from tensorflow import keras


def save_model(network, filename):
    """Saves an entire model.

    Args:
        network: model to save.
        filename: the path of the file that the model should be saved to.

    Returns:
        None.
    """
    network.save(filename)
    return None


def load_model(filename):
    """Loads an entire model.

    Args:
        filename: the path of the file that the model should be loaded from.

    Returns:
        the loaded model.
    """
    return keras.models.load_model(filename)

#!/usr/bin/env python3
"""
Weights
"""
import tensorflow.keras as K


def save_weights(network, filename, save_format='h5'):
    """Saves a model’s weights.

    Args:
        network: model whose weights should be saved.
        filename: the path of the file that the weights should be saved to.
        save_format: format in which the weights should be saved.

    Returns:
        None.
    """
    network.save_weights(filename, save_format=save_format)
    return None


def load_weights(network, filename):
    """Loads a model’s weights

    Args:
        network: model to which the weights should be loaded.
        filename: path of the file that the weights should be loaded from.

    Returns:
        None
    """
    network.load_weights(filename)

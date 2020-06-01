#!/usr/bin/env python3
"""
Config
"""

import tensorflow.keras as K


def save_config(network, filename):
    """Saves a model’s configuration in JSON format.

    Args:
        network: model whose configuration should be saved.
        filename: path of the file that the configuration should be saved to.

    Returns:
        None:
    """
    model_json = network.to_json()
    with open(filename, "w") as json_file:
        json_file.write(model_json)


def load_config(filename):
    """Loads a model with a specific configuration

    Args:
        filename: path of the file containing the model’s configuration in
                  JSON format.

    Returns:
        The loaded model.
    """
    with open(filename, "r") as json_file:
        loaded_model_json = json_file.read()

    loaded_model = K.models.model_from_json(loaded_model_json)

    return loaded_model

#!/usr/bin/env python3
"""
Prediction
"""
import tensorflow.keras as K


def predict(network, data, verbose=False):
    """Makes a prediction using a neural network.

    Args:
        network (Keras model): network model to make the prediction with.
        data (np.ndarray): input data to make the prediction with.
        verbose (bool): determines if output should be printed during the
                        prediction process.

    Returns:
        The prediction for the data.
    """
    return network.predict(data, verbose=verbose)

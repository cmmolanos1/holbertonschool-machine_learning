#!/usr/bin/env python3
"""
Test
"""
import tensorflow.keras as K


def test_model(network, data, labels, verbose=True):
    """

    Args:
        network (Keras model): the network model to test.
        data (np.ndarray): input data to test the model with.
        labels (np.ndarray): correct one-hot labels of data.
        verbose (bool): determines if output should be printed during the
                        testing process.

    Returns:
        float: the loss and accuracy of the model with the testing data,
               respectively
    """
    return network.evaluate(data, labels, verbose=verbose)

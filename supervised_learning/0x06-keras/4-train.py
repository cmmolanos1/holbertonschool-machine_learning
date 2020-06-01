#!/usr/bin/env python3
"""
Train
"""
from tensorflow import keras


def train_model(network, data, labels, batch_size, epochs, verbose=True,
                shuffle=False):
    """Trains a model using mini-batch gradient descent.

    Args:
        network (keras model): model to train.
        data (np.ndarray): matrix of shape (m, nx) containing the input data.
        labels (np.ndarray): one hot matrix of shape (m, classes) containing
                             the labels of data.
        batch_size (int): size of the batch used for mini-batch gradient
                          descent.
        epochs (int): number of passes through data for mini-batch gradient
                      descent.
        verbose (bool): determines if output should be printed during training.
        shuffle (bool): determines whether to shuffle the batches every epoch.
        Normally, it is a good idea to shuffle, but for reproducibility, we
        have chosen to set the default to False.

    Returns:
        The History object generated after training the model.

    """
    return network.fit(data, labels, batch_size, epochs,
                       verbose, shuffle)

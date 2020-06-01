#!/usr/bin/env python3
"""
Train
"""
import tensorflow.keras as K


def train_model(network, data, labels, batch_size, epochs,
                validation_data=None, early_stopping=False, patience=0,
                learning_rate_decay=False, alpha=0.1, decay_rate=1,
                save_best=False, filepath=None, verbose=True, shuffle=False):
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
        validation_data (tuple): data to validate the model with, if not None.
        early_stopping(bool): indicates whether early stopping should be used.
        patient (int): the patience used for early stopping.
        learning_rate_decay (bool): indicates whether learning rate decay
                                    should be used.
        alpha (float): learning rate.
        decay_rate (int): the decay rate.
        save_best (bool): indicates whether to save the model after each epoch
                          if it is the best.
        filepath (str): file path where the model should be saved.
        verbose (bool): determines if output should be printed during training.
        shuffle (bool): determines whether to shuffle the batches every epoch.
        Normally, it is a good idea to shuffle, but for reproducibility, we
        have chosen to set the default to False.

    Returns:
        The History object generated after training the model.

    """

    # learning rate schedule
    def step_decay(epoch):
        """Calculates the step decay"""
        return alpha / (1 + decay_rate * epoch)

    callbacks = []

    if validation_data and learning_rate_decay:
        callbacks.append(K.callbacks.LearningRateScheduler(step_decay,
                                                           verbose=1))

    if validation_data and early_stopping:
        callbacks.append(K.callbacks.EarlyStopping(monitor="val_loss",
                                                   patience=patience))

    if save_best:
        callbacks.append(K.callbacks.ModelCheckpoint(filepath,
                                                     save_best_only=True))

    return network.fit(x=data,
                       y=labels,
                       batch_size=batch_size,
                       epochs=epochs,
                       verbose=verbose,
                       validation_data=validation_data,
                       shuffle=shuffle,
                       callbacks=callbacks)

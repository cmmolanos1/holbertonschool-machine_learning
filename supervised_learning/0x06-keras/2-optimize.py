#!/usr/bin/env python3
"""
Optimize
"""
import tensorflow.keras as K


def optimize_model(network, alpha, beta1, beta2):
    """Sets up Adam optimization for a keras model with categorical
    crossentropy loss and accuracy metrics

    Args:
        network: the model to optimize.
        alpha: learning rate.
        beta1: first Adam optimization parameter.
        beta2: Second Adam optimization parameter.

    Returns:
        None
    """
    network.compile(optimizer=K.optimizers.Adam(lr=alpha,
                                                beta_1=beta1,
                                                beta_2=beta2),
                    loss='categorical_crossentropy',
                    metrics=['accuracy'])

    return None

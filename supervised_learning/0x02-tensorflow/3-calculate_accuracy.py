#!/usr/bin/env python3
"""Accuracy"""
import tensorflow as tf


def calculate_accuracy(y, y_pred):
    """Calculates the accuracy of a prediction

    Args:
        y (tensor): placeholder for the labels of the input data.
        y_pred (tensor): network’s predictions.

       Returns:
           tensor: the decimal accuracy of the prediction.

    """
    # We need to select the highest probability from the tensor that's
    # returned out of the softmax. One we have that, we compare it
    # against the actual value of y that we have should expected.
    correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_pred, 1))

    # Calculates and return the accuracy.
    return tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

#!/usr/bin/env python3
"""Create PlaceHolder"""
import tensorflow as tf


def create_placeholders(nx, classes):
    """Create 2 placeholders.

    Args:
        nx (int): the number of feature columns in our data.
        classes (int): number of classes in classifier.

    Returns:

    """
    x = tf.placeholder(tf.float32, shape=[None, nx], name="x")
    y = tf.placeholder(tf.float32, shape=[None, classes], name="y")

    return x, y

#!/usr/bin/env python3
"""
Pooling backward
"""

import tensorflow as tf


def lenet5(x, y):
    """Builds a modified version of the LeNet-5 architecture using tensorflow.

    Args:
        x (tf.placeholder): shape (m, 28, 28, 1) containing the input images
                            for the network.
        y (tf.placeholder): shape (m, 10) containing the one-hot labels for
                            the network.

    Returns:

    """
    init = tf.contrib.layers.variance_scaling_initializer()

    C1 = tf.layers.Conv2D(filters=6,
                          kernel_size=5,
                          padding='same',
                          kernel_initializer=init)(x)

    S2 = tf.layers.MaxPooling2D(pool_size=(2, 2),
                                strides=(2, 2))(C1)

    C3 = tf.layers.Conv2D(filters=16,
                          kernel_size=5,
                          padding='valid',
                          kernel_initializer=init)(S2)

    S4 = tf.layers.MaxPooling2D(pool_size=(2, 2),
                                strides=(2, 2))(C3)

    flatten = tf.layers.Flatten()(S4)

    C5 = tf.layers.Dense(units=120,
                         activation=tf.nn.relu,
                         kernel_initializer=init)(flatten)

    F6 = tf.layers.Dense(units=84,
                         activation=tf.nn.relu,
                         kernel_initializer=init)(C5)

    OUTPUT = tf.layers.Dense(units=10,
                             activation=None,
                             kernel_initializer=init)(F6)

    cost = tf.losses.softmax_cross_entropy(y, OUTPUT)

    correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(OUTPUT, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    optimizer = tf.train.AdamOptimizer().minimize(cost)

    return OUTPUT, optimizer, cost, accuracy

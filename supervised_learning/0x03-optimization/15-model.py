#!/usr/bin/env python3
"""
Batch normalization
"""

import numpy as np
import tensorflow as tf


def shuffle_data(X, Y):
    """Shuffles the data points in two matrices the same way.

    Args:
        X (np.ndarray): (m, nx) matrix to shuffle.
        Y (np.ndarray): (m, nx) matrix to shuffle.

    Returns:
        np.ndarray: shuffled version of X and Y.

    """
    m = X.shape[0]
    permutation = list(np.random.permutation(m))
    shuffled_X = X[permutation, :]
    shuffled_Y = Y[permutation, :]

    return shuffled_X, shuffled_Y


def create_layer(prev, n, activation):
    """ Create a NN layer.

    Args:
        prev (tensor): tensor output of the previous layer.
        n (int): number of nodes in the layer to create.
        activation (tf.nn.activation): activation function.

    Returns:
        tensor: the layer created with shape [?, n].

    """
    init = tf.contrib.layers.variance_scaling_initializer(mode="FAN_AVG")
    layer = tf.layers.Dense(units=n,
                            activation=activation,
                            kernel_initializer=init,
                            name="layer")

    return layer(prev)


def create_batch_norm_layer(prev, n, activation):
    """Creates a batch normalization layer for a neural network.

    Args:
        prev (tensor): the activated output of the previous layer.
        n (int): number of nodes in the layer to be created.
        activation (tensor): activation function that should be used on the
                             output of the layer

    Returns:
         tensor of the activated output for the layer.

    """
    # X, W, b ---> Z
    init = tf.contrib.layers.variance_scaling_initializer(mode="FAN_AVG")
    hidden = tf.layers.Dense(units=n, kernel_initializer=init)
    Z = hidden(prev)

    # Z, Gamma, Beta ---> Z_tilde
    epsilon = 1e-8
    beta = tf.Variable(tf.constant(0.0, shape=[n]), name='beta',
                       trainable=True)
    gamma = tf.Variable(tf.constant(1.0, shape=[n]), name='gamma',
                        trainable=True)
    mean, var = tf.nn.moments(Z, axes=[0])
    Z_tilde = tf.nn.batch_normalization(Z, mean, var, beta, gamma, epsilon)

    # A = g(Z_tilde)
    if activation is None:
        return Z_tilde
    return activation(Z_tilde)


def learning_rate_decay(alpha, decay_rate, global_step, decay_step):
    """Updates the learning rate using inverse time decay in numpy.

    Args:
        alpha (float): the original learning rate.
        decay_rate (float): the weight used to determine the rate at which
                            alpha will decay
        global_step (int): the number of passes of gradient descent that have
                           elapsed.
        decay_step (int): the number of passes of gradient descent that should
                          occur before alpha is decayed further.

    Returns:
        tensor: the learning rate decay operation
    """
    return tf.train.inverse_time_decay(alpha, global_step,
                                       decay_step, decay_rate,
                                       staircase=True)


def create_Adam_op(loss, alpha, beta1, beta2, epsilon):
    """Updates a variable in place using the Adam optimization algorithm.

    Args:
        loss (float):  loss of the network.
        alpha (float): learning rate.
        beta1 (float): weight used for the first moment.
        beta2 (float): weight used for the second moment.
        epsilon (float): small number to avoid division by zero.

    Returns:
        tensor: the Adam optimization operation
    """
    optimizer = tf.train.AdamOptimizer(alpha, beta1, beta2, epsilon)
    return optimizer.minimize(loss)


def forward_prop(x, layer_sizes=[], activations=[]):
    """Creates the forward propagation graph for the neural network.
    Normalizes all layers except the last one.
    Args:
        x (tensor): placeholder for the input data.
        layer_sizes (list): list containing the number of nodes in each layer
                            of the network.
        activations (list): list containing the activation functions for each
                            layer of the network.
    Returns:
        tensor: prediction of neural network.
    """
    A = create_batch_norm_layer(x, layer_sizes[0],
                                activations[0])
    for layer in range(1, len(layer_sizes)):
        if layer != len(layer_sizes) - 1:
            A = create_batch_norm_layer(A, layer_sizes[layer],
                                        activations[layer])
        else:
            A = create_layer(A, layer_sizes[layer],
                             activations[layer])
    return A


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


def calculate_loss(y, y_pred):
    """Calculates the cross-entropy loss of a prediction.

    Args:
        y (tensor): placeholder for the labels of the input data.
        y_pred (tensor): the network’s predictions.

    Returns:
        tensor: the loss of the prediction
    """
    return tf.losses.softmax_cross_entropy(y, y_pred)


def model(Data_train, Data_valid, layers, activations, alpha=0.001, beta1=0.9,
          beta2=0.999, epsilon=1e-8, decay_rate=1, batch_size=32, epochs=5,
          save_path='/tmp/model.ckpt'):
    """Builds, trains, and saves a neural network model in tensorflow using
    Adam optimization, mini-batch gradient descent, learning rate decay, and
    batch normalization.

    Args:
        Data_train (tuple): training inputs and training labels.
        Data_valid (tuple): validation inputs and validation labels.
        layers (list): number of nodes in each layer of the network.
        activations (list): activation functions used for each layer of the
                            network.
        alpha (float):  learning rate.
        beta1 (float): weight for the first moment of Adam Optimization.
        beta2 (float): weight for the second moment of Adam Optimization.
        epsilon (float): small number used to avoid division by zero.
        decay_rate (int): decay rate for inverse time decay of the learning
                          rate.
        batch_size (int): number of data points that should be in a mini-batch.
        epochs (int): number of times the training should pass through the
                      whole dataset.
        save_path (str): path where the model should be saved to.

    Returns:
        str: path where the model was saved.
    """
    m, nx = Data_train[0].shape
    classes = Data_train[1].shape[1]

    (X_train, Y_train) = Data_train
    (X_valid, Y_valid) = Data_valid

    x = tf.placeholder(tf.float32, shape=[None, nx], name="x")
    tf.add_to_collection('x', x)
    y = tf.placeholder(tf.float32, shape=[None, classes], name="y")
    tf.add_to_collection('y', y)

    y_pred = forward_prop(x, layers, activations)
    tf.add_to_collection('y_pred', y_pred)

    accuracy = calculate_accuracy(y, y_pred)
    tf.add_to_collection('accuracy', accuracy)

    loss = calculate_loss(y, y_pred)
    tf.add_to_collection('loss', loss)

    global_step = tf.Variable(0, trainable=False)
    alpha_d = learning_rate_decay(alpha, decay_rate, global_step, 1)

    train_op = create_Adam_op(loss, alpha_d, beta1, beta2, epsilon)
    tf.add_to_collection('train_op', train_op)

    init = tf.global_variables_initializer()
    saver = tf.train.Saver()

    m = X_train.shape[0]
    if m % batch_size == 0:
        complete = 1
        num_batches = int(m / batch_size)
    else:
        complete = 0
        num_batches = int(m / batch_size) + 1

    with tf.Session() as sess:
        sess.run(init)

        for i in range(epochs + 1):
            feed_t = {x: X_train, y: Y_train}
            feed_v = {x: X_valid, y: Y_valid}
            train_cost, train_accuracy = sess.run([loss, accuracy], feed_t)
            valid_cost, valid_accuracy = sess.run([loss, accuracy], feed_v)
            print("After {} epochs:".format(i))
            print("\tTraining Cost: {}".format(train_cost))
            print("\tTraining Accuracy: {}".format(train_accuracy))
            print("\tValidation Cost: {}".format(valid_cost))
            print("\tValidation Accuracy: {}".format(valid_accuracy))

            if i < epochs:
                X_shu, Y_shu = shuffle_data(X_train, Y_train)

                for k in range(num_batches):
                    if complete == 0 and k == num_batches - 1:
                        start = k * batch_size
                        X_minibatch = X_shu[start::]
                        Y_minibatch = Y_shu[start::]
                    else:
                        start = k * batch_size
                        end = (k * batch_size) + batch_size
                        X_minibatch = X_shu[start:end]
                        Y_minibatch = Y_shu[start:end]

                    feed_mb = {x: X_minibatch, y: Y_minibatch}
                    sess.run(train_op, feed_mb)

                    if (k + 1) % 100 == 0 and k != 0:
                        mb_c, mb_a = sess.run([loss, accuracy], feed_mb)
                        print("\tStep {}:".format(k + 1))
                        print("\t\tCost: {}".format(mb_c))
                        print("\t\tAccuracy: {}".format(mb_a))

            sess.run(tf.assign(global_step, global_step + 1))

        return saver.save(sess, save_path)

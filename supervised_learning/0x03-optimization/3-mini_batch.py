#!/usr/bin/env python3
"""
Train miniBatch
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


def train_mini_batch(X_train, Y_train, X_valid, Y_valid, batch_size=32,
                     epochs=5, load_path="/tmp/model.ckpt",
                     save_path="/tmp/model.ckpt"):
    """Trains a loaded neural network model using mini-batch gradient descent

    Args:
        X_train (np.ndarray): matrix (m, 784) containing the training data.
        Y_train (np.ndarray): matrix (m, 10) containing the training labels.
        X_valid (np.ndarray): matrix (m, 784) containing the validation data.
        Y_valid (np.ndarray): matrix (m, 10) containing the validation labels.
        batch_size (int): number of data points in a batch.
        epochs (int): number of times the training should pass through the
                      whole dataset.
        load_path (str): path from which to load the model.
        save_path (str): path to where the model should be saved after
                         training.

    Returns:
        str: the path where the model was saved.
    """
    m = X_train.shape[0]

    with tf.Session() as sess:
        saver = tf.train.import_meta_graph("{}.meta".format(load_path))
        saver.restore(sess, load_path)

        x = tf.get_collection("x")[0]
        y = tf.get_collection("y")[0]
        accuracy = tf.get_collection("accuracy")[0]
        loss = tf.get_collection("loss")[0]
        train_op = tf.get_collection("train_op")[0]

        for i in range(epochs + 1):
            # Print the train previous values
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
                mini_batches = []
                X_shu, Y_shu = shuffle_data(X_train, Y_train)
                num_complete_minibatches = int(m / batch_size)

                for k in range(0, num_complete_minibatches):
                    start = k * batch_size
                    end = (k * batch_size) + batch_size
                    mini_batch_X = X_shu[start: end, :]
                    mini_batch_Y = Y_shu[start: end, :]
                    mini_batch = (mini_batch_X, mini_batch_Y)
                    mini_batches.append(mini_batch)

                if m % batch_size != 0:
                    start = num_complete_minibatches * batch_size
                    mini_batch_X = X_shu[start::, :]
                    mini_batch_Y = Y_shu[start::, :]
                    mini_batch = (mini_batch_X, mini_batch_Y)
                    mini_batches.append(mini_batch)

                for mb in range(len(mini_batches)):
                    (minibatch_X, minibatch_Y) = mini_batches[mb]
                    sess.run(train_op, {x: minibatch_X, y: minibatch_Y})
                    mb_c, mb_a = sess.run([loss, accuracy],
                                          feed_dict={x: minibatch_X,
                                                     y: minibatch_Y})

                    if (mb + 1) % 100 == 0 and mb != 0:
                        print("\tStep {}:".format(mb + 1))
                        print("\t\tCost: {}".format(mb_c))
                        print("\t\tAccuracy: {}".format(mb_a))

        return saver.save(sess, save_path)

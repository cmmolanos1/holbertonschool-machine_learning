#!/usr/bin/env python3
"""evaluated"""

import tensorflow as tf


def evaluate(X, Y, save_path):
    """Evaluates the output of a neural network.

    Args:
        X (np.ndarray): input data to evaluate
        Y (np.ndarray): one-hot labels for X
        save_path (str): the location to load the model from

    Returns:
        float: network’s prediction.
        float: accuracy.
        float: loss

    """
    with tf.Session() as sess:
        saver = tf.train.import_meta_graph("{}.meta".format(save_path))
        saver.restore(sess, save_path)

        x = tf.get_collection("x")[0]
        y = tf.get_collection("y")[0]
        y_pred = tf.get_collection("y_pred")[0]
        accuracy = tf.get_collection("accuracy")[0]
        loss = tf.get_collection("loss")[0]

        y_pred_e, accuracy_e, loss_e = sess.run([y_pred, accuracy, loss],
                                                feed_dict={x: X, y: Y})

        return y_pred_e, accuracy_e, loss_e

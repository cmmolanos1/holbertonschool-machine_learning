#!/usr/bin/env python3
"""Train"""
import tensorflow as tf

calculate_accuracy = __import__('3-calculate_accuracy').calculate_accuracy
calculate_loss = __import__('4-calculate_loss').calculate_loss
create_placeholders = __import__('0-create_placeholders').create_placeholders
create_train_op = __import__('5-create_train_op').create_train_op
forward_prop = __import__('2-forward_prop').forward_prop


def train(X_train, Y_train, X_valid, Y_valid, layer_sizes, activations,
          alpha, iterations, save_path="/tmp/model.ckpt"):
    """Builds, trains, and saves a neural network classifier.

    Args:
        X_train (np.array): training input data.
        Y_train (np.array): training labels.
        X_valid (np.array): validation input data.
        Y_valid (np.array): validation labels.
        layer_sizes (list): list containing the number of nodes in each
                            layer of the network.
        activations (list): list containing the activation functions for each
                            layer of the network.
        alpha (float): learning rate.
        iterations (int): number of iterations to train over.
        save_path (str): where to save the model

    Returns:
        str: the path where the model was saved
    """
    m, nx = X_train.shape
    ny = Y_train.shape[1]

    x, y = create_placeholders(nx, ny)
    tf.add_to_collection('x', x)
    tf.add_to_collection('y', y)

    y_pred = forward_prop(x, layer_sizes, activations)
    tf.add_to_collection('y_pred', y_pred)

    accuracy = calculate_accuracy(y, y_pred)
    tf.add_to_collection('accuracy', accuracy)

    loss = calculate_loss(y, y_pred)
    tf.add_to_collection('loss', loss)

    optimizer = create_train_op(loss, alpha)
    tf.add_to_collection('optimizer', optimizer)

    init = tf.global_variables_initializer()
    saver = tf.train.Saver()

    with tf.Session() as sess:
        sess.run(init)
        for i in range(iterations + 1):

            cost_t, accuracy_t = sess.run([loss, accuracy],
                                          feed_dict={x: X_train, y: Y_train})
            cost_v, accuracy_v = sess.run([loss, accuracy],
                                          feed_dict={x: X_valid, y: Y_valid})

            if i % 100 == 0 or i == iterations:
                print("After {} iterations:".format(i))
                print("\tTraining Cost: {}".format(cost_t))
                print("\tTraining Accuracy: {}".format(accuracy_t))
                print("\tValidation Cost: {}".format(cost_v))
                print("\tValidation Accuracy: {}".format(accuracy_v))

            if i < iterations:
                sess.run(optimizer, feed_dict={x: X_train, y: Y_train})

        return saver.save(sess, save_path)

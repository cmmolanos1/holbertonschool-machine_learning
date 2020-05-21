#!/usr/bin/env python3
"""
Batch normalization
"""

import tensorflow as tf

shuffle_data = __import__('2-shuffle_data').shuffle_data
create_batch_norm_layer = __import__('14-batch_norm').create_batch_norm_layer
AdamOpt = __import__('10-Adam').create_Adam_op


def forward_prop(x, layer_sizes=[], activations=[]):
    """Creates the forward propagation graph for the neural network.
    Args:
        x (tensor): placeholder for the input data.
        layer_sizes (list): list containing the number of nodes in each layer
                            of the network.
        activations (list): list containing the activation functions for each
                            layer of the network.
    Returns:
        tensor: prediction of neural network.
    """
    for layer in range(0, len(layer_sizes)):
        if layer == 0:
            A = create_batch_norm_layer(x, layer_sizes[layer],
                                        activations[layer])
        else:
            A = create_batch_norm_layer(A, layer_sizes[layer],
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


def random_mini_batches(X_train, Y_train, batch_size):
    """Creates a tuples with minibatches

    Args:
        X_train (np.ndarray): Data input.
        Y_train (np.ndarray): Labels.
        batch_size (int): number of examples in a batch.

    Returns:
        list: (X, Y) * batches_number
    """
    m = X_train.shape[0]
    mini_batches = []
    X_shu, Y_shu = shuffle_data(X_train, Y_train)
    num_complete_minibatches = int(X_train.shape[0] / batch_size)

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

    return mini_batches


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
    alpha_d = tf.train.inverse_time_decay(alpha, global_step, 1, decay_rate,
                                          staircase=True)

    train_op = AdamOpt(loss, alpha_d, beta1, beta2, epsilon)
    tf.add_to_collection('train_op', train_op)

    init = tf.global_variables_initializer()
    saver = tf.train.Saver()

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
                minibatches = random_mini_batches(X_train, Y_train, batch_size)

                for mb in range(len(minibatches)):
                    (minibatch_X, minibatch_Y) = minibatches[mb]
                    sess.run(train_op, {x: minibatch_X, y: minibatch_Y})
                    mb_c, mb_a = sess.run([loss, accuracy],
                                          feed_dict={x: minibatch_X,
                                                     y: minibatch_Y})

                    if (mb + 1) % 100 == 0 and mb != 0:
                        print("\tStep {}:".format(mb + 1))
                        print("\t\tCost: {}".format(mb_c))
                        print("\t\tAccuracy: {}".format(mb_a))

            sess.run(tf.assign(global_step, global_step + 1))

        return saver.save(sess, save_path)

#!/usr/bin/env python3
"""Self Attention"""

import tensorflow as tf


class SelfAttention(tf.keras.layers.Layer):
    """
    Self attention class
    """

    def __init__(self, units):
        """Class constructor.

        Args:
            units (int):  the number of hidden units in the alignment model.
        """
        super(SelfAttention, self).__init__()
        self.W = tf.keras.layers.Dense(units)
        self.U = tf.keras.layers.Dense(units)
        self.V = tf.keras.layers.Dense(1)

    def call(self, s_prev, hidden_states):
        """

        Args:
            s_prev (tensor): of shape (batch, units) containing the previous
                             decoder hidden state.
            hidden_states (tensor): of shape (batch, input_seq_len, units)
                                    containing the outputs of the encoder.

        Returns:
            context, weights

            - context is a tensor of shape (batch, units) that contains the
              context vector for the decoder.
            - weights is a tensor of shape (batch, input_seq_len, 1) that
              contains the attention weights.
        """
        exp_s_prev = tf.expand_dims(s_prev, axis=1)
        score = self.V(tf.nn.tanh(self.W(exp_s_prev) + self.U(hidden_states)))
        weights = tf.nn.softmax(score, axis=1)
        context = tf.reduce_sum(weights * hidden_states, axis=1)

        return context, weights

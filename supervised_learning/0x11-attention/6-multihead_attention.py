#!/usr/bin/env python3
"""
Multihead attention
"""

import tensorflow as tf

sdp_attention = __import__('5-sdp_attention').sdp_attention


class MultiHeadAttention(tf.keras.layers.Layer):
    """
    Multiclass attention class
    """

    def __init__(self, dm, h):
        """Class constructor

        Args:
            dm (int): dimensionality of model.
            h (int): number of head
        """
        super(MultiHeadAttention, self).__init__()
        self.h = h
        self.dm = dm
        self.depth = int(self.dm / self.h)
        self.Wq = tf.keras.layers.Dense(self.dm)
        self.Wk = tf.keras.layers.Dense(self.dm)
        self.Wv = tf.keras.layers.Dense(self.dm)
        self.linear = tf.keras.layers.Dense(self.dm)

    def split_heads(self, x, batch_size):
        """Split the last dimension into (num_heads, depth).
        Transpose the result such that the shape is
        (batch_size, num_heads, seq_len, depth)
        """
        x = tf.reshape(x, (batch_size, -1, self.h, self.depth))
        return tf.transpose(x, perm=[0, 2, 1, 3])

    def call(self, Q, K, V, mask):
        """

        Args:
            Q (tensor): of shape (batch, seq_len_q, dk) containing the input
                        to generate the query matrix.
            K (tensor): of shape (batch, seq_len_v, dk) containing the input
                        to generate the key matrix.
            V (tensor): of shape (batch, seq_len_v, dv) containing the input
                        to generate the value matrix.
            mask: None

        Returns:
            output, weights

            - outputa tensor with its last two dimensions as
              (..., seq_len_q, dm) containing the scaled dot product attention.
            - weights a tensor with its last three dimensions as
              (..., h, seq_len_q, seq_len_v) containing the attention weights.
        """
        batch_size = tf.shape(Q)[0]

        Q = self.Wq(Q)
        K = self.Wk(K)
        V = self.Wv(V)

        q = self.split_heads(Q, batch_size)
        k = self.split_heads(K, batch_size)
        v = self.split_heads(V, batch_size)

        scaled_attention, weights = sdp_attention(q, k, v, mask)
        scaled_attention = tf.transpose(scaled_attention, perm=[0, 2, 1, 3])
        concat_attention = tf.reshape(scaled_attention,
                                      (batch_size, -1, self.dm))
        output = self.linear(concat_attention)

        return output, weights

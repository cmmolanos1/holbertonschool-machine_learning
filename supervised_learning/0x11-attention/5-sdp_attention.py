#!/usr/bin/env python3
"""
sdp-attention
"""

import tensorflow as tf


def sdp_attention(Q, K, V, mask=None):
    """calculates the scaled dot product attention.

    Args:
        Q (tensor): its last two dimensions as (..., seq_len_q, dk) containing
                    the query matrix.
        K (tensor):  its last two dimensions as (..., seq_len_v, dk)
                     containing the key matrix.
        V (tensor):  its last two dimensions as (..., seq_len_v, dv)
                     containing the value matrix.
        mask (tensor): is a tensor that can be broadcast into
                       (..., seq_len_q, seq_len_v) containing the optional
                       mask, or defaulted to None.

    Returns:
            output, weights
              - outputa tensor with its last two dimensions as
                (..., seq_len_q, dv) containing the scaled dot product
                attention.
              - weights a tensor with its last two dimensions as
              (..., seq_len_q, seq_len_v) containing the attention weights.
    """
    dk = tf.shape(Q)[-1]
    dk_float = tf.cast(dk, tf.float32)

    scaled = tf.matmul(Q, K, transpose_b=True) / tf.math.sqrt(dk_float)

    if mask is not None:
        scaled += (mask * -1e9)

    weights = tf.nn.softmax(scaled, axis=-1)
    output = tf.matmul(weights, V)

    return output, weights

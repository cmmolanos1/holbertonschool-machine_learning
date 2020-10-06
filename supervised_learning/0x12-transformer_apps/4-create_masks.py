#!/usr/bin/env python3
"""
Create mask
"""
import tensorflow.compat.v2 as tf


def create_masks(inputs, target):
    """creates all masks for training/validation.

    Args:
        inputs (tf.tensor): of shape (batch_size, seq_len_in) that contains
                            the input sentence.
        target (tf.tensor): of shape (batch_size, seq_len_out) that contains
                            the target sentence.

    Returns:
        encoder_mask, look_ahead_mask, decoder_mask
        -encoder_mask is the tf.Tensor padding mask of shape
        (batch_size, 1, 1, seq_len_in) to be applied in the encoder.
        -look_ahead_mask is the tf.Tensor look ahead mask of shape
        (batch_size, 1, seq_len_out, seq_len_out) to be applied in the decoder.
        -decoder_mask is the tf.Tensor padding mask of shape
        (batch_size, 1, 1, seq_len_in) to be applied in the decoder.
    """
    encoder_mask = tf.cast(tf.math.equal(inputs, 0), tf.float32)
    encoder_mask = encoder_mask[:, tf.newaxis, tf.newaxis, :]

    seq_len_out = target.shape[1]
    batch_size = target.shape[0]

    look_ahead_mask = \
        1 - tf.linalg.band_part(tf.ones((seq_len_out, seq_len_out)), -1, 0)
    look_ahead_mask = \
        tf.repeat(look_ahead_mask[tf.newaxis, tf.newaxis, :, :],
                  batch_size, axis=0)

    decoder_mask = tf.cast(tf.math.equal(inputs, 0), tf.float32)
    decoder_mask = decoder_mask[:, tf.newaxis, tf.newaxis, :]

    return encoder_mask, look_ahead_mask, decoder_mask

#!/usr/bin/env python3
"""Transformer Decoder"""

import tensorflow as tf

positional_encoding = __import__('4-positional_encoding').positional_encoding
DecoderBlock = __import__('8-transformer_decoder_block').DecoderBlock


class Decoder(tf.keras.layers.Layer):
    """
    Decorder class
    """

    def __init__(self, N, dm, h, hidden, target_vocab, max_seq_len,
                 drop_rate=0.1):
        """Class constructor

        Args:
            N (int): the number of blocks in the encoder.
            dm (int): the dimensionality of the model.
            h (int): the number of heads.
            hidden (int): the number of hidden units in the fully connected
                          layer.
            target_vocab (int): the size of the target vocabulary.
            max_seq_len (int): the maximum sequence length possible.
            drop_rate (float): the float rate
        """
        super(Decoder, self).__init__()
        self.N = N
        self.dm = dm
        self.embedding = tf.keras.layers.Embedding(target_vocab, self.dm)
        self.positional_encoding = positional_encoding(max_seq_len, self.dm)
        self.blocks = [DecoderBlock(self.dm, h, hidden, drop_rate)
                       for n in range(self.N)]
        self.dropout = tf.keras.layers.Dropout(drop_rate)

    def call(self, x, encoder_output, training,
             look_ahead_mask, padding_mask):
        """

        Args:
            x (tensor): of shape (batch, target_seq_len, dm)containing the
                        input to the decoder.
            encoder_output (tensor): of shape (batch, input_seq_len, dm)
                                     containing the output of the encoder.
            training (bool): determine if the model is training.
            look_ahead_mask: the mask to be applied to the first multi head
                             attention layer.
            padding_mask: the mask to be applied to the second multi head
                          attention layer.

        Returns:
            a tensor of shape (batch, target_seq_len, dm) containing the
            decoder output.
        """
        seq_len = x.shape[1]

        x = self.embedding(x)
        x *= tf.math.sqrt(tf.cast(self.dm, tf.float32))
        x += self.positional_encoding[:seq_len]

        X = self.dropout(x, training=training)

        for n in range(self.N):
            X = self.blocks[n](X, encoder_output, training,
                               look_ahead_mask, padding_mask)
        return X

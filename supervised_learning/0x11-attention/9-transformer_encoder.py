#!/usr/bin/env python3
"""
Transformer Encoder
"""

import tensorflow as tf

positional_encoding = __import__('4-positional_encoding').positional_encoding
EncoderBlock = __import__('7-transformer_encoder_block').EncoderBlock


class Encoder(tf.keras.layers.Layer):
    """
    Encoder Class
    """

    def __init__(self, N, dm, h, hidden, input_vocab, max_seq_len,
                 drop_rate=0.1):
        """Class constructor

        Args:
            N (int): the number of blocks in the encoder.
            dm (int): the dimensionality of the model.
            h (int): the number of heads.
            hidden (int): the number of hidden units in the fully connected
                          layer.
            input_vocab (int): the size of the input vocabulary.
            max_seq_len (int):  the maximum sequence length possible.
            drop_rate (float): the dropout rate
        """
        super(Encoder, self).__init__()
        self.N = N
        self.dm = dm
        self.embedding = tf.keras.layers.Embedding(input_vocab, dm)
        self.positional_encoding = positional_encoding(max_seq_len, self.dm)
        self.blocks = [EncoderBlock(dm, h, hidden, drop_rate)
                       for n in range(self.N)]
        self.dropout = tf.keras.layers.Dropout(drop_rate)

    def call(self, x, training, mask):
        """

        Args:
            x (tensor): of shape (batch, input_seq_len, dm)containing the
                        input to the encoder.
            training (bool): determine if the model is training.
            mask: the mask to be applied for multi head attention.

        Returns:
            a tensor of shape (batch, input_seq_len, dm) containing the
            encoder output.
        """
        seq_len = x.shape[1]

        x = self.embedding(x)
        x *= tf.math.sqrt(tf.cast(self.dm, tf.float32))
        x += self.positional_encoding[:seq_len]

        X = self.dropout(x, training=training)
        for n in range(self.N):
            X = self.blocks[n](X, training, mask)
        return X

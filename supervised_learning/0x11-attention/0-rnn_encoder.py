#!/usr/bin/env python3
"""
RNN Encoder
"""

import tensorflow as tf


class RNNEncoder(tf.keras.layers.Layer):
    """ RNN Encoder class"""

    def __init__(self, vocab, embedding, units, batch):
        """Class constructor.

        Args:
            vocab (int): the size of the input vocabulary.
            embedding (int): dimensionality of the embedding vector.
            units (int): number of hidden units in the RNN cell.
            batch (int): the batch size.
        """
        super(RNNEncoder, self).__init__()
        self.batch = batch
        self.units = units
        self.embedding = tf.keras.layers.Embedding(vocab, embedding)
        self.gru = tf.keras.layers.GRU(units,
                                       recurrent_initializer="glorot_uniform",
                                       return_sequences=True,
                                       return_state=True)

    def initialize_hidden_state(self):
        """Initializes the hidden states for the RNN cell to a tensor of
        zeros.

        Returns:
            a tensor of shape (batch, units)containing the initialized hidden
            states.
        """
        initializer = tf.keras.initializers.Zeros()
        init_hidden_state = initializer(shape=(self.batch, self.units))
        return init_hidden_state

    def call(self, x, initial):
        """

        Args:
            x (tensor): of shape batch, input_seq_len) containing the input to
                        the encoder layer as word indices within the
                        vocabulary.
            initial (tensor): of shape (batch, units) containing the initial
                              hidden state.

        Returns:
            outputs, hidden

            - outputs is a tensor of shape (batch, input_seq_len, units)
              containing the outputs of the encoder.
            - hidden is a tensor of shape (batch, units) containing the last
              hidden state of the encoder.
        """
        embedding = self.embedding(x)
        full_seq_outputs, last_hidden_state = self.gru(embedding,
                                                       initial_state=initial)
        return full_seq_outputs, last_hidden_state

#!/usr/bin/env python3

"""
Datasets
"""
import tensorflow.compat.v2 as tf
import tensorflow_datasets as tfds


class Dataset:
    """
    Class Dataset
    """

    def __init__(self):
        """
        Class constructor
        """
        self.data_train = tfds.load('ted_hrlr_translate/pt_to_en',
                                    split=['train'],
                                    as_supervised=True)
        self.data_valid = tfds.load('ted_hrlr_translate/pt_to_en',
                                    split=['validation'],
                                    as_supervised=True)

        self.tokenizer_pt, self.tokenizer_en = \
            self.tokenize_dataset(self.data_train)

    def tokenize_dataset(self, data):
        """Creates sub-word tokenizers for our dataset.

        Args:
            data (tf.data.Dataset): whose examples are formatted as a tuple
            (pt, en)
            - pt is the tf.Tensor containing the Portuguese sentence
            - en is the tf.Tensor containing the corresponding English sentence

        Returns:
            tokenizer_pt, tokenizer_en
            - tokenizer_pt is the Portuguese tokenizer
            -tokenizer_en is the English tokenizer
        """
        tokenizer_pt = tfds.features.text.SubwordTextEncoder.build_from_corpus(
            (en.numpy() for pt, en in data), target_vocab_size=2 ** 15)

        tokenizer_en = tfds.features.text.SubwordTextEncoder.build_from_corpus(
            (pt.numpy() for pt, en in data), target_vocab_size=2 ** 15)

        return tokenizer_pt, tokenizer_en

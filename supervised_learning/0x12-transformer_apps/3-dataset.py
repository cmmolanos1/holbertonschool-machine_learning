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

    def __init__(self, batch_size, max_len):
        """Class constructor

        Args:
            batch_size (int): the batch size for training/validation.
            max_len (int): the batch size for training/validation.
        """
        examples, metadata = tfds.load('ted_hrlr_translate/pt_to_en',
                                       with_info=True,
                                       as_supervised=True)
        self.data_train, self.data_valid = examples['train'], \
            examples['validation']

        self.tokenizer_pt, self.tokenizer_en = \
            self.tokenize_dataset(self.data_train)

        self.data_train = self.data_train.map(self.tf_encode)
        self.data_valid = self.data_valid.map(self.tf_encode)

        def filter_max_length(x, y, max_length=max_len):
            return tf.logical_and(tf.size(x) <= max_length,
                                  tf.size(y) <= max_length)

        # Train
        self.data_train = self.data_train.filter(filter_max_length)
        # cache the dataset to memory to get a speedup while reading from it.
        self.data_train = self.data_train.cache()
        train_dataset_size = metadata.splits['train'].num_examples
        self.data_train = self.data_train.shuffle(train_dataset_size)
        padded_shapes = ([None], [None])
        self.data_train = \
            self.data_train.padded_batch(batch_size,
                                         padded_shapes=padded_shapes)
        self.data_train = \
            self.data_train.prefetch(tf.data.experimental.AUTOTUNE)

        # Valid
        self.data_valid = self.data_valid.filter(filter_max_length)
        self.data_valid = \
            self.data_valid.padded_batch(batch_size,
                                         padded_shapes=padded_shapes)

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
            (pt.numpy() for pt, en in data), target_vocab_size=2 ** 15)

        tokenizer_en = tfds.features.text.SubwordTextEncoder.build_from_corpus(
            (en.numpy() for pt, en in data), target_vocab_size=2 ** 15)

        return tokenizer_pt, tokenizer_en

    def encode(self, pt, en):
        """encodes a translation into tokens

        Args:
            pt (tf.Tensor): containing the Portuguese sentence.
            en (tf.Tensor): containing the corresponding English sentence.

        Returns:
            pt_tokens, en_tokens
            - pt_tokens is a tf.Tensor containing the Portuguese tokens
            - en_tokens is a tf.Tensor containing the English tokens
        """
        pt_tokens = [self.tokenizer_pt.vocab_size] + self.tokenizer_pt.encode(
            pt.numpy()) + [self.tokenizer_pt.vocab_size + 1]

        en_tokens = [self.tokenizer_en.vocab_size] + self.tokenizer_en.encode(
            en.numpy()) + [self.tokenizer_en.vocab_size + 1]

        return pt_tokens, en_tokens

    def tf_encode(self, pt, en):
        """Acts as a tensorflow wrapper for the encode instance method.

        Args:
            pt (tf.Tensor): containing the Portuguese sentence.
            en (tf.Tensor): containing the corresponding English sentence.

        Returns:
            pt, en: tensors.
        """
        result_pt, result_en = tf.py_function(self.encode,
                                              [pt, en],
                                              [tf.int64, tf.int64])
        result_pt.set_shape([None])
        result_en.set_shape([None])

        return result_pt, result_en

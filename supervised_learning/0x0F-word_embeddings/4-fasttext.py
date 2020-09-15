#!/usr/bin/env python3
"""Fasttext"""

import gensim


def fasttext_model(sentences, size=100, min_count=5, negative=5, window=5,
                   cbow=True, iterations=5, seed=0, workers=1):
    """creates and trains a genism fastText model.

    Args:
        sentences (list): sentences to be trained on.
        size (int): the dimensionality of the embedding layer.
        min_count (int): the minimum number of occurrences of a word for use
                         in training.
        negative (int): the size of negative sampling.
        window (int): the maximum distance between the current and predicted
                      word within a sentence.
        cbow (bool): is a boolean to determine the training type; True is for
                     CBOW; False is for Skip-gram.
        iterations (int): the number of iterations to train over.
        seed (int): the seed for the random number generator.
        workers (int): the number of worker threads to train the model.

    Returns:
        The trained model.
    """
    model = gensim.models.FastText(sentences,
                                   min_count=min_count,
                                   iter=iterations,
                                   size=size,
                                   window=window,
                                   sg=cbow,
                                   seed=seed,
                                   negative=negative)

    model.train(sentences,
                total_examples=model.corpus_count,
                epochs=model.iter)

    return model

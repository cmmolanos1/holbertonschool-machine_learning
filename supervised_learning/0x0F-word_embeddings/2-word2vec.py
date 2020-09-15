#!/usr/bin/env python3
"""Word2vec"""

import gensim


def word2vec_model(sentences, size=100, min_count=5, window=5, negative=5,
                   cbow=True, iterations=5, seed=0, workers=1):
    """creates and trains a gensim word2vec model.

    Args:
        sentences(list): sentences to be trained on.
        size (int): the dimensionality of the embedding layer.
        min_count (int): the minimum number of occurrences of a word for use
                         in training.
        window (int): the maximum distance between the current and predicted
                      word within a sentence.
        negative (int): the size of negative sampling.
        cbow (bool): boolean to determine the training type; True is for CBOW;
                     False is for Skip-gram.
        iterations (int): the number of iterations to train over.
        seed (int): the seed for the random number generator.
        workers (int): the number of worker threads to train the model.
    Returns:
        the trained model.
    """
    model = gensim.models.Word2Vec(sentences,
                                   min_count=min_count,
                                   iter=iterations,
                                   size=size,
                                   window=window,
                                   sg=cbow,
                                   seed=seed,
                                   negative=negative,
                                   workers=workers)

    model.train(sentences,
                total_examples=model.corpus_count,
                epochs=model.iter)

    return model

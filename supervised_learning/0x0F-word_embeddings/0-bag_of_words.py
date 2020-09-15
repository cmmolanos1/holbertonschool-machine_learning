#!/usr/bin/env python3

from sklearn.feature_extraction.text import CountVectorizer


def bag_of_words(sentences, vocab=None):
    """reates a bag of words embedding matrix.

    Args:
        sentences (list): sentences to analize.
        vocab (list):  vocabulary words to use for the analysis.

    Returns:
        embeddings, features
        - embeddings is a numpy.ndarray of shape (s, f) containing the
          embeddings.
            s is the number of sentences in sentences
            f is the number of features analyzed
        - features is a list of the features used for embeddings
    """
    if vocab is None:
        vectorizer = CountVectorizer()
        X = vectorizer.fit_transform(sentences)
        vocab = vectorizer.get_feature_names()

    else:
        vectorizer = CountVectorizer(vocabulary=vocab)
        X = vectorizer.fit_transform(sentences)

    embedding = X.toarray()

    return embedding, vocab

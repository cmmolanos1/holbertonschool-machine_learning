#!/usr/bin/env python3
"""
bleu
"""

import numpy as np


def ngram_bleu(references, sentence, n):
    """calculates the n-gram BLEU score for a sentence.

    Args:
        references (list): references translations. each reference translation
                           is a list of the words in the translation.
        sentence (list):  list containing the model proposed sentence.
        n (int): the size of the n-gram to use for evaluation.

    Returns:
        The n-gram BLEU score.
    """
    output_len = len(sentence)
    count_clip = 0
    references_len = []
    counts_clip = {}

    # n-sentence pass to the grams that we need
    n_sentence = [' '.join([str(j) for j in sentence[i:i + n]])
                  for i in range(len(sentence) - (n - 1))]

    # n_sentence = [(str(sentence[i]) + ' ' + str(sentence[i+1]))
    #              for i in range(len(sentence)-(n-1))]
    n_output_len = len(n_sentence)

    for reference in references:
        n_reference = [' '.join([str(j) for j in reference[i:i + n]])
                       for i in range(len(sentence) - (n - 1))]

        references_len.append(len(reference))
        for word in n_reference:
            if word in n_sentence:
                if not counts_clip.keys() == word:
                    counts_clip[word] = 1

    count_clip = sum(counts_clip.values())

    reference_len = min(references_len, key=lambda x: abs(x - output_len))

    if output_len > reference_len:
        bp = 1
    else:
        bp = np.exp(1 - (reference_len / output_len))

    bleu_score = bp * np.exp(np.log(count_clip / n_output_len))

    return bleu_score

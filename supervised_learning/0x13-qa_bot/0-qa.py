#!/usr/bin/env python3
"""
Question answer
"""
import tensorflow as tf
import tensorflow_hub as hub
from transformers import BertTokenizer


def question_answer(question, reference):
    """finds a snippet of text within a reference document to answer a
    question.

    Args:
        question (str): question to answer.
        reference (str): the reference document from which to find the answer.

    Returns:

    """
    tokenizer = BertTokenizer.from_pretrained('bert-large-uncased-whole-word-'
                                              'masking-finetuned-squad')
    model = hub.load("https://tfhub.dev/see--/bert-uncased-tf2-qa/1")

    question = question
    paragraph = reference

    question_tokens = tokenizer.tokenize(question)
    paragraph_tokens = tokenizer.tokenize(paragraph)

    tokens = ['[CLS]'] + question_tokens + ['[SEP]'] + paragraph_tokens + \
             ['[SEP]']

    input_word_ids = tokenizer.convert_tokens_to_ids(tokens)
    input_mask = [1] * len(input_word_ids)
    input_type_ids = [0] * (1 + len(question_tokens) + 1) + \
                     [1] * (len(paragraph_tokens) + 1)

    input_word_ids, input_mask, input_type_ids = \
        map(lambda t: tf.expand_dims(
            tf.convert_to_tensor(t, dtype=tf.int32), 0),
            (input_word_ids, input_mask, input_type_ids))

    outputs = model([input_word_ids, input_mask, input_type_ids])

    short_start = tf.argmax(outputs[0][0][1:]) + 1
    short_end = tf.argmax(outputs[1][0][1:]) + 1

    answer_tokens = tokens[short_start: short_end + 1]
    answer = tokenizer.convert_tokens_to_string(answer_tokens)

    return answer

import numpy as np
import re

re_punc = r'^[^a-zA-Z0-9]+$'
re_hash = r'^#'
re_at = r'^@'
re_num = r'\d'  # just remove all words with numbers
re_url = r'(^http:|\.com$)'


def clean_word(w):
    """
    Cleans a word by normalizing it and replacing some with special tokens.
    """
    w = w.strip()
    if re.match(re_punc, w):
        return '#PUNC#'
    if re.match(re_hash, w):
        return '#HASH#'
    if re.match(re_at, w):
        return '#AT#'
    if re.match(re_num, w):
        return '#NUM#'
    if re.match(re_url, w):
        return '#URL#'
    return w.lower()


def get_token_mapping(observations, min_freq=0):
    """
    Converts a dictionary of words into a vocabulary mapping.
    
    Arguments:
        observations - Dictionary where keys are words, and values are their counts.
        min_freq - Mininum word count for a word to be added to vocabulary.
        
    Return:
        token_mapping - Dictionary where keys are words in a vocabulary,
            and values are a unique integer assigned to the word.
    """
    token_mapping = {}
    for word in observations:
        if observations[word] > min_freq:
            token_mapping[word] = len(token_mapping)
    token_mapping['#UNK#'] = len(token_mapping)
    return token_mapping


def tokenize(token_mapping, sentence):
    """
    Function to convert each word into its corresponding integer.
    If word is not found, return the value for '#UNK'.
    """
    return [token_mapping.get(w, token_mapping['#UNK#']) for w in sentence]

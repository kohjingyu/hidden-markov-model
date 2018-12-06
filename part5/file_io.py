import os
from collections import defaultdict, OrderedDict
from preprocess import clean_word


def parse(train_filename):
    """
    Reads a file and counts all word and label occurrences within it.
    
    Arguments:
        train_filename - Path to input file (e.g. data/EN/train)
    
    Return:
        observations - Dictionary containing all words and corresponding counts
        states - Dictionary containing all labels and corresponding counts
    """
    with open(train_filename, 'r', encoding="utf-8") as f:
        lines = f.readlines()

    observations, states = defaultdict(int), defaultdict(int)
    for line in lines:
        data_split = line.strip().rsplit(' ', 1)
        if len(data_split) == 2:
            obs, state = data_split
            observations[clean_word(obs)] += 1
            states[state] += 1
    return observations, states


def read_file(filename, clean=True, test=False):
    """
    Reads a file and returns the sentence and corresponding labels in it.
    
    Arguments:
        filename - Path to input file (e.g. data/EN/dev.in)
        clean - Performs word cleaning if true
        
    Return:
        sentences - List of sentences in file, where each sentence is a list of words
        labels - List of list of labels in file, corresponding to sentences
    """
    with open(filename, 'r', encoding="utf-8") as f:
        lines = f.readlines()
        
    # Write original sentences instead of cleaned sentences for test data
    sentence, original_sentence, label = [], [], []
    sentences, labels = [], []
    original_sentences = []
    for line in lines:
        data_split = line.strip().rsplit(' ', 1)
        if test and len(data_split) == 1 and data_split[0] != '':
            word = data_split[0]
            original_sentence.append(word)
            if clean:
                word = clean_word(word)
            sentence.append(word)
        elif len(data_split) == 2:
            word, state = data_split
            if clean:
                word = clean_word(word)
            sentence.append(word)
            label.append(state)
        else:
            sentences.append(sentence)
            original_sentences.append(original_sentence)
            labels.append(label)
            sentence = []
            original_sentence = []
            label = []

    if test:
        return sentences, original_sentences, labels
    else:
        return sentences, labels

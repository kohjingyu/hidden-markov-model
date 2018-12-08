import os
import numpy as np
import pandas as pd
from collections import defaultdict
import re
from tqdm import tqdm as tqdm, tnrange as trange

def clean_word(w):
    '''
    1. Converts stopwords into #STOP#. Our list of stopwords is derived from the nltk library.
    2. Converts punctuations, #s, @s, numbers, URLs into accompanying genericized labels
    '''
    stopwords = {'it', 'before', "you're", 'wouldn', "hasn't", 'again', 'just', "you'd", 'only', "wouldn't", "isn't", 'no', 'yourself', 'them', 'in', 's', "aren't", 'because', 'ourselves', 'more', "couldn't", "shan't", "shouldn't", 'shan', 't', 'needn', "hadn't", 'both', 'such', 'below', 'out', 'my', "haven't", 'there', 'off', 'few', 'against', 'each', 'can', 'yourselves', 'some', 'further', 'if', 'under', 'over', 'theirs', 'his', 'herself', 'been', 'her', 'than', "you've", 'into', 'have', 'aren', 'and', 'then', 'whom', 'having', 'the', 'by', 'itself', 'him', 'up', 'be', "wasn't", 'me', 'of', 'haven', 'they', "mightn't", 'our', 'are', 'how', "weren't", 'couldn', 'too', "should've", 'until', 'weren', "that'll", 'a', 'when', "doesn't", 'which', 'here', 'was', 'hers', 'ain', 'mightn', 'those', 'himself', 'or', 'while', 'i', 'yours', 'from', 'once', 'after', 'does', 'its', 'to', 'you', 'myself', 'don', 'shouldn', 'most', 'that', 'mustn', 'same', "it's", 'what', 'any', 'did', 'isn', 'she', 'these', 'with', 'during', "needn't", 'your', 'their', 'being', 'on', 'above', 've', 'won', 'down', "she's", 're', 'this', 'do', 'hasn', 'had', 'so', 'other', 'an', 'were', 'hadn', 'themselves', 'will', 'as', 'for', 'at', 'd', 'm', 'but', 'ma', 'didn', 'll', 'not', "won't", 'ours', 'who', 'through', 'all', 'now', 'o', 'nor', "mustn't", "you'll", 'where', 'doesn', 'y', 'doing', 'why', 'between', "didn't", 'he', 'we', 'wasn', 'has', 'very', "don't", 'own', 'about', 'is', 'should', 'am'}
    re_punc = r'^[^a-zA-Z0-9]+$'
    re_hash = r'^#'
    re_at = r'^@'
    re_num = r'\d'  # just remove all words with numbers
    re_url = r'(^http:|\.com$)'
    w = w.strip()
    if w in stopwords:
        return '#STOP#'
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

dictionary = set()

# The behavior of the emissions matrix is the same - we use the same function as used in Part 3
def learn_emissions(train_filename):
    ''' Learns emissions parameters from data and returns them as a nested dictionary '''
    with open(train_filename, "r") as f:
        lines = f.readlines()

    observations = set()
    # Track emission counts
    emissions = {} # Where key is y, and value is a dictionary of emissions x from y with their frequency

    # Learn from data
    for line in tqdm(lines, desc='Emissions'):
        data_split = line.strip().rsplit(' ', 1)

        # Only process valid lines
        if len(data_split) == 2:
            obs, state = data_split
            obs = clean_word(obs)
            observations.add(obs)

            # Track this emission
            if state in emissions:
                current_emissions = emissions[state]
            else:
                current_emissions = defaultdict(int)

            if obs not in dictionary:
                dictionary.add(obs)
                current_emissions["#UNK#"] += 1

            current_emissions[obs] += 1
            emissions[state] = current_emissions # Update

    emission_counts = {k: sum(emissions[k].values()) for k in emissions}
    return emissions, emission_counts, observations


def get_emission_parameters(emissions, emission_counts, observations, x, y, k=1):
    ''' Returns the MLE of the emission parameters based on the emissions dictionary '''
    if y not in emissions:  # edge case: no records of emission from this state
        return 0
    state_data = emissions[y]
    count_y = emission_counts[y] #sum(state_data.values()) # Denominator

    if x not in observations:  # edge case: no records of emission of this word
        count_y_x = k
    else:
        count_y_x = state_data[x]

    e = count_y_x / (count_y + k)
    return e


def learn_transitions(train_filename):
    """
    Returns a dictionary containing (key, value) where
        key: (t, u, v)
        value: Count(t, u, v)
    """
    with open(train_filename, 'r') as f:
        lines = f.readlines()

    transitions = defaultdict(int)
    prev_prev_state = 'PRESTART'
    prev_state = 'START'
    # avoid excessive indentations
    for line in tqdm(lines, desc='Transitions'):
        data_split = line.strip().rsplit(' ', 1)

        # line breaks -> new sequence
        if len(data_split) < 2:
            transitions[(prev_prev_state, prev_state, 'STOP')] += 1
            prev_prev_state = 'PRESTART'
            prev_state = 'START'
            continue

        obs, curr_state = data_split
        transitions[(prev_prev_state, prev_state, curr_state)] += 1
        prev_prev_state = prev_state
        prev_state = curr_state

    # count number of 'from' states
    transition_counts = defaultdict(int)
    for (t, u, v), counts in transitions.items():
        transition_counts[(t, u)] += counts

    # get all unique states
    t, u, v = zip(*transitions)
    states = set(t) | set(u) | set(v)
    return transitions, transition_counts, states

def get_transition_parameters(transitions, transition_counts, t, u, v):
    if transition_counts[(t, u)] == 0:  # edge case: no records of transitions starting from (t, u)
        return 0
    return transitions[(t, u, v)] / transition_counts[(t, u)]


def viterbi(transitions, transition_counts, states, emissions, emission_counts, observations, obs_seq):
    a = lambda prev2, prev, curr: get_transition_parameters(transitions, transition_counts, prev2, prev, curr)
    b = lambda state, out: get_emission_parameters(emissions, emission_counts, observations, y=state, x=out)

    # create empty tables
    n = len(obs_seq) + 2  # (PRESTART + START), (START, obs_seq[0]) + ... + (obs_seq[n], STOP)

    state_combis = set()
    for u in states:
        for v in states:
            state_combis.add((u,v))
    P = pd.DataFrame(index=state_combis, columns=range(n)).fillna(0)  # probability table
    B = pd.DataFrame(index=state_combis, columns=range(n))  # backpointer table

    # initialization
    P.loc[('PRESTART', 'START'), 0] = 1

    # forward recursion
    sentence = []
    for j in range(1, n-1):
        x = clean_word(obs_seq[j-1])
        sentence.append(x)
        if x not in dictionary:
            x = "#UNK#"
        for v in states:  # curr state
            for u in states:  # prev state
                for t in states: # prev_2 state
                    p = P.loc[(t, u), j-1] * a(t, u, v) * b(v, x)
                    if p > P.loc[(u, v), j]:
                        P.loc[(u, v), j] = p  # update probability
                        B.loc[(u, v), j] = t  # update backpointer - t is the grandfather
    print(' '.join(sentence))

    # termination
    j = n - 1
    v = 'STOP'
    for u in states: # prev state
        for t in states: # prev_2 state
            p = P.loc[(t, u), j-1] * a(t, u, v)
            if p > P.loc[(u, v), j]:
                P.loc[(u, v), j] = p  # probability
                B.loc[(u, v), j] = t  # backpointer

    # backtracing
    # second order viterbi requires backpropagation keep track of the second order item,
    # as the limited horizon assumption is now limited to *two* entries instead.
    state_pair = P[n-1].idxmax()
    state_seq = []
    # References next row in the backpointer table using the next state pair
    # I.e. Child, Parent -> Grandparent
    # then assign Child = Parent
    # and Parent = Grandparent
    # then Grandparent = Next Grandparent via our backpointer table
    for i in range(n-1, 0, -1):
        # Grab the next state according to our backpointer table
        next_state = B.loc[state_pair, i]
        if isinstance(next_state, str):
            state_seq.append(state_pair[1])
            state_pair = (next_state, state_pair[0])
        else: # edge case: no possible transition to START
            for j in range(int(i)):
                state_seq.append(('O'))
            break
    state_seq = state_seq[::-1][:-1]  # reverse and drop STOP
    print(state_seq)
    return P, B, state_seq

def train(dataset):
    """
    Takes in dataset name, obtains transition and emission parameters.
    """
    train_filename = f'../data/{dataset}/train'
    # train
    t = learn_transitions(train_filename)
    e = learn_emissions(train_filename)
    return t, e

def validate(dataset, t, e):
    """
    Takes in dataset name, and HMM parameters.
    Use HMM parameters to estimate state sequence.
    Write state sequence to file alongside input observation sequence.
    """
    val_filename = f'../data/{dataset}/dev.in'
    out_filename = f'../data/{dataset}/dev.p4.out'

    # validate with train parameters
    label_sequence = lambda sentence: viterbi(*t, *e, sentence)

    # read validation file for sequence
    with open(val_filename, 'r') as f:
        lines = f.readlines()

    sentence = []
    result = []
    for word in tqdm(lines, desc=dataset):
        if word == '\n':
            print(sentence)
            _, _, s = label_sequence(sentence)  # ignore tables
            result.extend(s)
            result.append('\n')
            sentence = []
        else:
            sentence.append(word.strip())

    # write sequence to out file
    with open(out_filename, 'w') as f:
        for i in range(len(lines)):
            word = lines[i].strip()
            if word:
                pred = result[i]
                f.write(word + ' ' + pred)
            f.write('\n')

datasets = ['EN', 'FR']
for dataset in datasets:
    t, e = train(dataset)
    validate(dataset, t, e)

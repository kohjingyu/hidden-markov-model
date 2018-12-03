#!/usr/bin/env python
# coding: utf-8
import os
import numpy as np
import pandas as pd
from collections import defaultdict
from tqdm import tqdm, trange
from IPython.display import display


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
            observations.add(obs)

            # Track this emission
            if state in emissions:
                current_emissions = emissions[state]
            else:
                current_emissions = defaultdict(int)
                
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
        key: (u, v)
        value: Count(u, v)
    """
    with open(train_filename, 'r') as f:
        lines = f.readlines()
        
    transitions = defaultdict(int)
    prev_state = 'START'
    # avoid excessive indentations
    for line in tqdm(lines, desc='Transitions'):
        data_split = line.strip().rsplit(' ', 1)
        
        # line breaks -> new sequence
        if len(data_split) < 2:
            transitions[(prev_state, 'STOP')] += 1
            prev_state = 'START'
            continue

        obs, curr_state = data_split
        transitions[(prev_state, curr_state)] += 1
        prev_state = curr_state
        
    # count number of 'from' states
    transition_counts = defaultdict(int)
    for (u, v), counts in transitions.items():
        transition_counts[u] += counts

    # get all unique states
    u, v = zip(*transitions)
    states = set(u) | set(v)
    return transitions, transition_counts, states


def get_transition_parameters(transitions, transition_counts, u, v):
    if transition_counts[u] == 0:  # edge case: no records of transitions starting from u
        return 0
    return transitions[(u, v)] / transition_counts[u]


def viterbi(transitions, transition_counts, states, emissions, emission_counts, observations, obs_seq):
    a = lambda prev, curr: get_transition_parameters(transitions, transition_counts, prev, curr)
    b = lambda state, out: get_emission_parameters(emissions, emission_counts, observations, x=out, y=state)

    # create empty tables
    n = len(obs_seq) + 2  # START + |obs_seq| + STOP
    P = pd.DataFrame(index=states, columns=range(n)).fillna(0)  # probability table
    B = pd.DataFrame(index=states, columns=range(n))  # backpointer table
    
    # initialization
    P.loc['START', 0] = 1
    
    # recursion
    for j in range(1, n-1):
        x = obs_seq[j-1]  # obs_seq starts from 0, j starts from 1
        for v in states:  # curr state
            for u in states:  # prev state
                p = P.loc[u, j-1] * a(u, v) * b(v, x)
                if p > P.loc[v, j]:
                    P.loc[v, j] = p  # update probability
                    B.loc[v, j] = u  # update backpointer
                    
    # termination
    j = n - 1
    v = 'STOP'
    for u in states:
        p = P.loc[u, j-1] * a(u, v)
        if p > P.loc[v, j]:
            P.loc[v, j] = p  # probability
            B.loc[v, j] = u  # backpointer
            
    # backtrace
    state_seq = ['STOP']
    for i in range(n-1, 0, -1):
        curr_state = state_seq[-1]
        prev_state = B.loc[curr_state, i]
        if pd.isnull(prev_state):  # edge case: no possible transition to STOP
            state_seq = ['O'] * n
            break
        state_seq.append(prev_state)
    state_seq = state_seq[::-1][1:-1]  # reverse and drop START, STOP
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
    out_filename = f'../data/{dataset}/dev.p3.out'

    # validate with train parameters
    label_sequence = lambda sentence: viterbi(*t, *e, sentence)
    
    # read validation file for sequence
    with open(val_filename, 'r') as f:
        lines = f.readlines()

    sentence = []
    result = []
    for word in tqdm(lines, desc=dataset):
        if word == '\n':
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


if __name__ == '__main__':
    datasets = ['SG', 'CN', 'EN', 'FR']
    for dataset in datasets:
        print(dataset)
        t, e = train(dataset)
        validate(dataset, t, e)

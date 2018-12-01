
# coding: utf-8

# # Part 2
# In this section, we write some dank functions for the HMM
import numpy as np
from tqdm import tqdm

import sys


def learn_emissions(train_filename):
    ''' Learns emissions parameters from data and returns them as a nested dictionary '''
    with open(train_filename, "r") as f:
        lines = f.readlines()

#     # Keep set of all unique states and observations
#     states = set()
    observations = set()

    # Track emission counts
    emissions = {} # Where key is y, and value is a dictionary of emissions x from y with their frequency

    # Learn from data
    for line in lines:
        data_split = line.strip().split(" ")

        # Only process valid lines
        if len(data_split) == 2:
            obs, state = data_split[0], data_split[1]

#             states.add(state)
            observations.add(obs)

            # Track this emission
            current_emissions = {}
            if state in emissions:
                current_emissions = emissions[state]

            # If it exists, increment it, if not set it to 1
            if obs in current_emissions:
                current_emissions[obs] += 1
            else:
                current_emissions[obs] = 1

            emissions[state] = current_emissions # Update
    
    emission_counts = {k: sum(emissions[k].values()) for k in emissions}
    
    return emissions, emission_counts, observations


# ## Estimating Emission Parameters
# We make use of MLE to estimate the emission parameters based on the training data.
def get_emission_parameters(emissions, x, y):
    ''' Returns the MLE of the emission parameters based on the emissions dictionary '''
    state_data = emissions[y]
    count_y_x = state_data[x] # Numerator
    count_y = sum(state_data.values()) # Denominator
    
    e = count_y_x / count_y
    return e


# ## Estimating with Smoothing
def get_emission_parameters(emissions, emission_counts, x, y, k=1):
    ''' Returns the MLE of the emission parameters based on the emissions dictionary '''
    state_data = emissions[y]
    count_y = emission_counts[y] #sum(state_data.values()) # Denominator
    
    # If x == "#UNK#", it will return the following
    count_y_x = k
    
    # If x exists in training, return its MLE instead
    if x != "#UNK#":
        count_y_x = state_data[x] # Numerator
    
    e = count_y_x / (count_y + k)
    return e

def label_sequence(sentence, emissions, emission_counts):
    ''' Takes a list `sentence` that contains words of a sentence as strings '''
    tags = []
    
    for word in sentence:
        predicted_label = ""
        max_prob = -1
        
        # Find y with maximum probability
        for y in emissions:
            
            if word not in observations:
                word = "#UNK#"
            
            if (word in emissions[y]) or (word == "#UNK#"):
                prob = get_emission_parameters(emissions, emission_counts, word, y)
            
                # If this is higher than the previous highest, use this
                if prob > max_prob:
                    predicted_label = y
                    max_prob = prob

        # Add prediction to list
        tags.append(predicted_label)
    
    return tags

# # Training and Validation

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python hmm_part2.py SG")

    dataset = sys.argv[1]
    assert(dataset in ["SG", "CN", "EN", "FR"])

    print("Processing for {}".format(dataset))

    train_filename = f"data/{dataset}/train"
    validation_filename = f"data/{dataset}/dev.in"

    # Train
    emissions, emission_counts, observations = learn_emissions(train_filename)


    # Test on validation set
    with open(validation_filename, "r") as f:
        lines = f.readlines()
        sentence = []

    result = []

    for word in tqdm(lines):
        # If it's a newline, it's the end of a sentence. Predict!
        if word == "\n":
            preds = label_sequence(sentence, emissions, emission_counts)
            
            # Add predictions to overall results
            result += preds
            result += ["\n"]
            
            # Reset sentence list
            sentence = []
        else: # Sentence has not ended
            # Add word to sentence
            sentence.append(word.strip())


    # Write predictions to file
    with open(validation_filename.replace(".in", ".p2.out"), "w") as f:
        for i in range(len(lines)):
            word = lines[i].strip()
            
            # Only write if it's not a newline
            if word:
                pred = result[i]
                f.write(word + " " + pred)
            
            f.write("\n")


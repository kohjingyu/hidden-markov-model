# Hidden Markov Models

Authors: Gabriel Wong (1002299), Lee Tze How (1002033), Koh Jing Yu (1002045)

This repository hosts the code for our submission for the 01.112 Project. It contains 4 parts: the 3 sections as described in the project description, and our design project (`part5`).

For a full documentation of our methods and results, please refer to our report (`report/report.pdf`).

## Directory Setup

All training and validation data is placed in the root of this repository after cloning. The file structure should be as follows:

```
data
   - CN
   - EN
   - FR
   - SG
```

## Part 2

Part 2 entails estimating the emission probabilities from the given dataset. Our code is contained within `part2/hmm_part2.py`. To run it for a particular dataset, execute the following on the command line:

```
python hmm_part2.py SG
```

Where `SG` may be replaced by any of the valid datasets. After running the script, all predictions are written to the appropriate dataset directory (for example, `data/SG/dev.p2.out`).

## Part 3

Part 3 involves estimating the transition probabilities from the given dataset, and implementing the Viterbi algorithm to compute the most likely sequence. To run our code, execute the following:

```
python hmm_part3.py
```

The emission parameters are estimated using the same code as in part 2. The script performs parameter estimation and Viterbi on all 4 datasets, and writes the results to the appropriate file.

## Part 4

Part 4 asks us to implement a second order Hidden Markov Model. 

## Evaluation script
Pass no arguments to run evaluation on all.
```
./eval.sh [dataset] [qn]
# example
./eval.sh  # eval all
./eval.sh EN 2
./eval.sh SG 3
```

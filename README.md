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

## Design Challenge (Part 5)

For our design challenge, we implemented and evaluated several deep neural networks. Our most successful model is using a multilayer perceptron. We implement forward pass and backpropagation from scratch using `numpy`. For more information, please refer to our report.

To train the MLP on the appropriate dataset (`FR` or `EN`), we accept several command line arguments:

```
usage: mlp.py [-h] [--epochs EPOCHS] [--batch BATCH] [--lr LR]
              [--hidden HIDDEN] [--decay DECAY] [--dropout DROPOUT]
              dataset

positional arguments:
  dataset            {EN, FR}

optional arguments:
  -h, --help         show this help message and exit
  --epochs EPOCHS    No. of epochs [10]
  --batch BATCH      Batch size [64]
  --lr LR            Learning rate [1e-2]
  --hidden HIDDEN    No. of hidden units [128]
  --decay DECAY      Learning rate decay [1.0]
  --dropout DROPOUT  Probability of dropping [0.0]
```

To replicate our results, please follow the training procedure as detailed in our report.

## Evaluation script
Pass no arguments to run evaluation on all.
```
./eval.sh [dataset] [qn]
# example
./eval.sh  # eval all
./eval.sh EN 2
./eval.sh SG 3
```

## Acknowledgements

This project was jointly worked on by Gabriel Wong, Lee Tze How, and Koh Jing Yu from SUTD. We thank Prof. Lu Wei and our TA Ngo Van Mao for their valuable advice and help during the course.
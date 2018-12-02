# Hidden Markov Models

## Directory Setup

All training and validation data is placed in the root of this repository after cloning. The file structure should be as follows:

```
data
   - CN
   - EN
   - FR
   - SG
```

## Evaluation script
Pass no arguments to run evaluation on all.
```
./eval.sh [dataset] [qn]
# example
./eval.sh  # eval all
./eval.sh EN 2
./eval.sh SG 3
```

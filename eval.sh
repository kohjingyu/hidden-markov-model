#!/bin/bash
if [ -z "$1" ]
then
    echo EN
    echo Part 2
    python3.6 evalResult.py data/EN/dev.out data/EN/dev.p2.out | grep F
    echo Part 3
    python3.6 evalResult.py data/EN/dev.out data/EN/dev.p3.out | grep F
    echo Part 4
    python3.6 evalResult.py data/EN/dev.out data/EN/dev.p4.out | grep F
    echo Part 5
    python3.6 evalResult.py data/EN/dev.out data/EN/dev.p5.out | grep F

    echo SG
    echo Part 2
    python3.6 evalResult.py data/SG/dev.out data/SG/dev.p2.out | grep F
    echo Part 3
    python3.6 evalResult.py data/SG/dev.out data/SG/dev.p3.out | grep F

    echo CN
    echo Part 2
    python3.6 evalResult.py data/CN/dev.out data/CN/dev.p2.out | grep F
    echo Part 3
    python3.6 evalResult.py data/CN/dev.out data/CN/dev.p3.out | grep F

    echo FR
    echo Part 2
    python3.6 evalResult.py data/FR/dev.out data/FR/dev.p2.out | grep F
    echo Part 3
    python3.6 evalResult.py data/FR/dev.out data/FR/dev.p3.out | grep F
    echo Part 4
    python3.6 evalResult.py data/FR/dev.out data/FR/dev.p4.out | grep F
    echo Part 5
    python3.6 evalResult.py data/FR/dev.out data/FR/dev.p5.out | grep F
else
    python3.6 evalResult.py data/$1/dev.out data/$1/dev.p$2.out
fi

#/bin/bash

#PBS -q normal
#PBS -l select=1:ncpus=4:mem=16GB
#PBS -l walltime=24:00:00

#PBS -N "HMM"
#PBS -P Personal

cd /home/users/sutd/1002045/HMM/part5
#module load python/3.5.1
source activate py3k

declare -a datasets=("EN" "FR")
declare -a batches=("8" "16" "32" "50" "64")
declare -a lrs=("1e-2" "1e-3" "1e-4")
declare -a hiddens=("32" "64" "128" "150" "256" "512")
declare -a decay=("1.0" "0.95" "0.9")
declare -a dropout=("0.0" "0.2" "0.3" "0.5")

for dataset in "${datasets[@]}"
do
    for batch in "${batches[@]}"
    do
        for lr in "${lrs[@]}"
        do
            for hidden in "${hiddens[@]}"
            do
                for d in "${decay[@]}"
                do
                    for do in "${dropout[@]}"
                    do
                       python3 mlp.py ${dataset} --batch ${batch} --lr ${lr} --hidden ${hidden} --decay ${d} --dropout ${do} &> logs/${dataset}_batch${batch}_lr${lr}_decay${d}_hidden${hidden}_dropout${do}.txt
                    done
                done
            done
        done
    done
done

#!/bin/bash

model=mamba2
eval_dataset='15min-part'

for dataset in 4000 2000; do 
    for seed in 53 97 103 191 99719; do # 0 17 1243 3674 7341 53 97 103 191 99719
        echo $model
        echo $dataset
        echo $eval_dataset
        echo $seed
        python evaluate.py --baseline -m $model -s1 $dataset -d $eval_dataset --seed $seed > evaluate/baseline/full/$eval_dataset-$model-$dataset-$seed.txt 
    done
done
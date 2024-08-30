#!/bin/bash

model=mamba2
eval_dataset='15min-full'

for dataset in 4000 6000 8000; do 
    for seed in 53 97 103 191 99719; do # 0 17 1243 3674 7341 53 97 103 191 99719
        echo $model
        echo $dataset
        echo $eval_dataset
        echo $seed
        python evaluate.py --baseline -m $model -s1 $dataset -d $eval_dataset --seed $seed > evaluate/baseline/logs/$eval_dataset-$model-$dataset-$seed.txt 
    done
done
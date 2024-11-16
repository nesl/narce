#!/bin/bash

model=soft_ae_transformer

for eval_dataset in '30min'; do #'3min' '15min' '30min' '5min'
    for dataset in 4000; do # 100 200 400 500 600 800 1000 2000 4000 6000 8000 10000
        for seed in 0 17 1243 3674 7341 53 97 103 191 99719; do # 0 17 1243 3674 7341 53 97 103 191 99719
            echo $model
            echo $dataset
            echo $eval_dataset
            echo $seed
            mkdir -p evaluate/baseline/logs/$model
            python evaluate.py --baseline -m $model -s1 $dataset -d $eval_dataset --seed $seed > evaluate/baseline/logs/$model/$eval_dataset-$model-$dataset-$seed.txt 
        done
    done
done
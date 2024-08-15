#!/bin/bash

model=mamba2

for dataset in 4000 2000; do 
    for seed in 53 97 103 191 99719; do # 0 17 1243 3674 7341 53 97 103 191 99719
        echo $model
        echo $dataset
        echo $seed
        python baseline.py $model $dataset $seed > baseline/saved_logs/full/$model-$dataset-$seed.txt 
    done
done

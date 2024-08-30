#!/bin/bash

model=transformer

for dataset in 2000 4000 6000 8000 10000; do 
    for seed in 53 97 103 191 99719; do # 0 17 1243 3674 7341 53 97 103 191 99719
        echo $model
        echo $dataset
        echo $seed
        python baseline.py $model $dataset $seed > baseline/saved_logs/$model-$dataset-$seed.txt 
    done
done

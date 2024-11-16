#!/bin/bash

model=soft_ae_mamba1

for dataset in 1000 2000 4000 6000 8000 10000; do # 1000 2000 4000 6000 8000 10000
    for seed in 0 17 1243 3674 7341 53 97 103 191 99719; do # 0 17 1243 3674 7341 53 97 103 191 99719
        echo $model
        echo $dataset
        echo $seed
        mkdir -p baseline/saved_logs/$model
        python baseline.py $model $dataset $seed > baseline/saved_logs/$model/$model-$dataset-$seed.txt 
    done
done

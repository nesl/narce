#!/bin/bash

model=narce_mamba2_6L
nar_dataset=40000
eval_dataset='5min-full'

for sensor_dataset in 2000 4000; do
    for seed in 53 97 103 191 99719; do # 0 17 1243 3674 7341 53 97 103 191 99719
        echo $model
        echo $nar_dataset
        echo $sensor_dataset
        echo $eval_dataset
        echo $seed
        python evaluate.py --narce -m $model -s1 $sensor_dataset -s2 $nar_dataset -d $eval_dataset --seed $seed > evaluate/narce/$eval_dataset-$model-$nar_dataset-$sensor_dataset-$seed.txt 
    done
done


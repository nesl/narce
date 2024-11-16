#!/bin/bash

model=narce_mlp_1L
nar_dataset=20000
# eval_dataset='5min'

for eval_dataset in '30min'; do #'3min' '15min' '30min' '5min'
    for sensor_dataset in 2000; do
        for seed in 99719; do # 0 17 1243 3674 7341 53 97 103 191 99719
            echo $model
            echo $nar_dataset
            echo $sensor_dataset
            echo $eval_dataset
            echo $seed
            mkdir -p evaluate/narce/logs/$model/n_$nar_dataset/s_$sensor_dataset
            python evaluate.py --narce -m $model -s1 $sensor_dataset -s2 $nar_dataset -d $eval_dataset --seed $seed > evaluate/narce/logs/$model/n_$nar_dataset/s_$sensor_dataset/$eval_dataset-$model-$nar_dataset-$sensor_dataset-$seed.txt 
        done
    done
done

#!/bin/bash

model=nar_mamba2
train=adapter
nar_dataset=10000
for sensor_dataset in 2000; do
    for seed in 53 97 103; do # 53 97 103 191 99719; do
        echo $model
        echo $train
        echo $nar_dataset
        echo $sensor_dataset
        echo $seed
        python train_nar.py $model $train $nar_dataset $sensor_dataset $seed > nar/saved_logs/NAR_adapter/adapter-$model-$nar_dataset-$sensor_dataset-$seed.txt 
    done
done

#!/bin/bash

nar_model=mamba1_v1
adapter_model=mamba1_12L
train=pipeline
nar_dataset=20000
for sensor_dataset in 2000; do
    for seed in 53 97 103 191 99719; do # 53 97 103 191 99719; do
        echo $nar_model
        echo $adapter_model
        echo $train
        echo $nar_dataset
        echo $sensor_dataset
        echo $seed
        mkdir -p narce/saved_logs/$train
        python narce.py $nar_model $adapter_model $train $nar_dataset $sensor_dataset $seed > narce/saved_logs/$train/$nar_model-$nar_dataset-$adapter_model-$sensor_dataset-$seed.txt 
    done
done

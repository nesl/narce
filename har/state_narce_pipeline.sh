#!/bin/bash

nar_model=state_mamba2_v1
adapter_model=mamba2_12L
train=pipeline

nar_dataset=10000
for sensor_dataset in 4000 2000; do
    for seed in 53 97 103; do # 53 97 103 191 99719; do
        echo $nar_model
        echo $adapter_model
        echo $train
        echo $nar_dataset
        echo $sensor_dataset
        echo $seed
        python narce.py $nar_model $adapter_model $train $nar_dataset $sensor_dataset $seed > narce/saved_logs/$train/$nar_model-$nar_dataset-$adapter_model-$sensor_dataset-$seed.txt 
    done
done

#!/bin/bash

nar_model=mamba1_v1
adapter_model=None # this does not matter
sensor_dataset=2000
train=nar
nar_dataset=20000
for seed in 53; do # 53 97 103 191 99719; do
    echo $nar_model
    echo $adapter_model
    echo $train
    echo $nar_dataset
    echo $sensor_dataset
    echo $seed
    mkdir -p narce/saved_logs/$train
    python narce.py $nar_model $adapter_model $train $nar_dataset $sensor_dataset $seed > narce/saved_logs/$train/$nar_model-$nar_dataset-$seed.txt 
done

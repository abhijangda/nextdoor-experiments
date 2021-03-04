#!/bin/bash


source venv/bin/activate
echo $(python train.py --custom_data $1 --nomultilabel --num_layers 3 --num_clusters 1500 --bsize 20 --hidden1 512 --dropout 0.2 --weight_decay 0  --early_stopping 200 --num_clusters_val 20 --num_clusters_test 1 --epochs 1 --save_name $1  --learning_rate 0.005 --diag_lambda 0.0001 --novalidation)

deactivate

#!/bin/bash

source venv/bin/activate
# sample_method = fastgcn | ladies
echo $(python3 pytorch_ladies.py --cuda 0 --dataset $1 --sample_method $2)
deactivate 

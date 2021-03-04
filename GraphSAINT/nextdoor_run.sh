#!/bin/bash



source venv/bin/activate
rm -f models/*
echo $(python -m graphsaint.tensorflow_version.train --data_prefix  $1  --train_config train_config/mrw.yml --gpu 0) 
deactivate


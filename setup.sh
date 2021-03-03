#!/bin/bash


cd GraphSAINT
virtualenv venv
source venv/bin/activate
pip install -r requirements.txt
cd ..
echo "Setup GraphSAINT Complete"

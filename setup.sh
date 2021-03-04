#!/bin/bash


cd GraphSAINT
virtualenv venv
chmod +x nextdoor_run.sh
source venv/bin/activate
pip install -r requirements.txt
cd ..
echo "Setup GraphSAINT Complete"

cd cluster_gcn
virtualenv venv
chmod +x run_custom.sh
source venv/bin/activate
pip install -r requirements.txt
cd ..
echo "Setup ClusterGCN complete"


cd LADIES
virtualenv venv
chmod +x run_custom.sh
source venv/bin/activate
pip install -r requrements.txt
cd ..
echo "Setup FastGCN|LADIES Complete"

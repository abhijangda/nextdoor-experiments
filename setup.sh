#!/bin/bash


#cd GraphSAINT
#virtualenv venv
#chmod +x GraphSAINT/nextdoor_run.sh
#source venv/bin/activate
#pip install -r requirements.txt
#cd ..
#echo "Setup GraphSAINT Complete"

cd cluster_gcn
virtualenv venv
chmod +x cluster_gcn/run_custom.sh
source venv/bin/activate
pip install -r requirements.txt
cd ..
echo "Setup ClusterGCN complete"


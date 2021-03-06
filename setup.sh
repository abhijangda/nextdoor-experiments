#!/bin/bash


cd graph_loading
chmod +x build.sh
./build.sh
cd ..
echo "Graph loading module built"


cd GraphSAGE
virtualenv venv
source venv/bin/activate
pip install -r requirements.txt
cd ..
echo "Setup GraphSAGE Complete"


cd GraphSAINT
virtualenv venv
chmod +x nextdoor_run.sh
source venv/bin/activate
pip install -r requirements.txt
python graphsaint/setup.py build_ext --inplace
cd ..
echo "Setup GraphSAINT Complete!"

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
pip install -r requirements.txt
cd ..
echo "Setup FastGCN|LADIES Complete"

import os
import sys

USAGE = "USAGE:python  test_all_gnn.py (GNN_NAME) (DATASET) \
        GNN_NAME = graphsaint \
        DATASET = ppi | patents | orkut | livejournal | reddit "

if len(sys.argv) != 3:
    print(USAGE)

import subprocess 
import re

gnn = sys.argv[1]
dataset = sys.argv[2]

if gnn == 'graphsaint':
    os.system("rm -f GraphSAINT/*.pickle")
    os.system("rm -f GraphSAINT/models/*")
    process = subprocess.Popen(["./nextdoor_run.sh",dataset],cwd="GraphSAINT",stdout = subprocess.PIPE) 
    output = process.communicate()[0]
    output = output.decode('utf-8')
    training_time = re.search("training_time: (\d+\.\d+)",output).groups()[0]
    sampling_time = re.search("sampling_time: (\d+\.\d+)",output).groups()[0]
    
if gnn == 'cluster_gcn':
    #os.system("cat cluster_gcn/run_custom.sh")
    process = subprocess.Popen(["./run_custom.sh",dataset],cwd="cluster_gcn",stdout = subprocess.PIPE)
    output = process.communicate()[0]
    output = output.decode('utf-8')
    #print(output)
    training_time = re.search("training_time: (\d+\.\d+)",output).groups()[0]
    sampling_time = re.search("sampling_time: (\d+\.\d+)",output).groups()[0]

if gnn == 'fastgcn' or gnn == 'ladies':
    if gnn == 'fastgcn':
        sample_method = 'fastgcn'
    else:
        sample_method = 'ladies'
    process = subprocess.Popen(["./run_custom.sh",dataset,sample_method],cwd="LADIES",stdout = subprocess.PIPE)
    output = process.communicate()[0]
    output = output.decode('utf-8')
    print(output)
    training_time = re.search("training_time: (\d+\.\d+)",output).groups()[0]
    sampling_time = re.search("sampling_time: (\d+\.\d+)",output).groups()[0]








print("Training time",training_time)
print("Sampling time", sampling_time)

import os
import sys

USAGE = "USAGE:python  test_all_gnn.py (GNN_NAME) (ROOT_DIR) (DATASET) \
                python test_all_gnn.py (ROOT_DIR) \
        GNN_NAME = graphsaint | cluster_gcn |  \
        DATASET = ppi | patents | orkut | livejournal | reddit "

if len(sys.argv) != 4 or len(sys.argv) != 2:
    print(USAGE)

import subprocess 
import re

#gnn = sys.argv[1]
#dataset = sys.argv[2]

def run_graph_saint(dataset):
    try:
        os.system("rm -f GraphSAINT/*.pickle")
        os.system("rm -f GraphSAINT/models/*")
        process = subprocess.Popen(["./nextdoor_run.sh",dataset],cwd="GraphSAINT",stdout = subprocess.PIPE) 
        output = process.communicate()[0]
        output = output.decode('utf-8')
        training_time = re.search("training_time: (\d+\.\d+)",output).groups()[0]
        sampling_time = re.search("sampling_time: (\d+\.\d+)",output).groups()[0]
        return training_time,sampling_time
    except:
        return "OOM","OOM"

def run_cluster_gcn(dataset):
    try:
        #os.system("cat cluster_gcn/run_custom.sh")
        process = subprocess.Popen(["./run_custom.sh",dataset],cwd="cluster_gcn",stdout = subprocess.PIPE)
        output = process.communicate()[0]
        output = output.decode('utf-8')
        #print(output)
        training_time = re.search("training_time: (\d+\.\d+)",output).groups()[0]
        sampling_time = re.search("sampling_time: (\d+\.\d+)",output).groups()[0]
        return training_time,sampling_time
    except:
        return "OOM","OOM"

def run_fastgcn_or_ladies(gnn, dataset):
    if gnn == 'fastgcn':
        sample_method = 'fastgcn'
    else:
        sample_method = 'ladies'
    try:    
        process = subprocess.Popen(["./run_custom.sh",dataset,sample_method],cwd="LADIES",stdout = subprocess.PIPE)
        output = process.communicate()[0]
        output = output.decode('utf-8')
        print(output)
        training_time = re.search("training_time: (\d+\.\d+)",output).groups()[0]
        sampling_time = re.search("sampling_time: (\d+\.\d+)",output).groups()[0]
        return training_time,sampling_time
    except:
        return "OOM","OOM"

def run_mvs_gcn(dataset):
    try:
        process = subprocess.Popen(["./run_custom.sh",dataset],cwd="mvs_gcn",stdout = subprocess.PIPE)
        output = process.communicate()[0]
        output = output.decode('utf-8')
        #print(output)
        training_time = re.search("training_time: (\d+\.\d+)",output).groups()[0]
        sampling_time = re.search("sampling_time: (\d+\.\d+)",output).groups()[0]
        return training_time,sampling_time
    except:
        return "OOM","OOM"

def run_graphsage(root_dir,dataset):
    if True:
        os.chdir('./GraphSAGE')
        gnnCommand = "python3 experiment/epoch_run_time.py {} {}"
        status,output = subprocess.getstatusoutput("env -i bash -c 'source venv/bin/activate && env'")
        for line in output.split('\n'):
            (key, _, value) = line.partition("=")
            os.environ[key] = value
    
        c = gnnCommand.format(root_dir,dataset)
        print(c)
        status,output = subprocess.getstatusoutput(c)
        print(output)
        training_time = re.search("training_time: (\d+\.\d+)",output).groups()[0]
        sampling_time = re.search("sampling_time: (\d+\.\d+)",output).groups()[0]
        return training_time,sampling_time
    #except:
        return "OOM","OOM"


def run_everything():
    results = {}
    # format ("gnn arch:(dataset, training_time, sampling_time)")
    lamdas = {'graphsage':run_graphsage, 'graphsaint':run_graph_saint, 'cluster_gcn':run_cluster_gcn,\
            'ladies': lambda dataset: run_fastgcn_or_ladies('ladies',dataset), \
             'fastgcn': lambda dataset: run_fastgcn_or_ladies('fastgcn',dataset)}
    dataset = ["ppi","patents","orkut","livejournal","reddit"]
    for gnn in lambdas.keys():
        results[gnn] = []
        for d in dataset:
           t,s = lambdas(d)
           results[gnn].append(d,t,s)
    print("GNN Arch | Dataset | training_time | sampling_time")
    for arc in results.keys():
        for time in results[arc]:
            print("{} | {} | {} | {}".format(arc,time[0], time[1], time[2]))


def run_gnn_and_dataset(gnn,rootdir,dataset):
    if gnn == 'graphsage':
        t,s  = run_graphsage(rootdir,dataset)
    elif gnn == 'graphsaint':
        t,s = run_graph_saint(dataset)
    elif gnn == 'cluster_gcn':
        t,s = run_cluster_gcn(dataset)
    elif gnn =='ladies' or gnn == 'fastgcn':
        t,s = run_fastgcn_or_ladies(gnn,dataset)
    elif gnn == 'mvs_gcn':
        t,s = run_mvs_gcn(dataset)
    print(t,s) 


if __name__=="__main__":
    print(len(sys.argv))
    if len(sys.argv) < 2:
        run_everything()
    else:
        if len(sys.argv) !=4:
            print(USAGE)
        else:
            gnn = sys.argv[1]
            root_dir = sys.argv[2]
            dataset = sys.argv[3]
            run_gnn_and_dataset(gnn,root_dir,dataset)

#print("Training time",training_time)
#print("Sampling time", sampling_time)

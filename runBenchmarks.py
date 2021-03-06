import argparse
import os
import subprocess
import re
import datetime 

#TODO: Store the output as log some where.
logFile = os.path.join(os.getcwd(), "benchmarking.log")

parser = argparse.ArgumentParser(description='Benchmark')
parser.add_argument('-knightKing', type=str,
                    help='Path to KnightKing',required=True)
parser.add_argument('-nextdoor', type=str,
                    help='Path to NextDoor',required=True)
# parser.add_argument('-runs', type=int, help="Number of Runs",required=True)
parser.add_argument('-gpus', type=str, help="CUDA DEVICES",required=False)

args = parser.parse_args()
args.nextdoor = os.path.abspath(args.nextdoor)
args.knightKing = os.path.abspath(args.knightKing)
cwd = os.getcwd()

input_dir = os.path.join(args.nextdoor, "input")

#Run KnightKing Benchmarks
graphInfo = {
    "PPI": {"v": 56944, "path": os.path.join(input_dir, "ppi.data")},
    "LiveJournal": {"v": 4576926, "path": os.path.join(input_dir, "LJ1.data")},
    "Orkut": {"v":3072441,"path":os.path.join(input_dir, "orkut.data")},
    "Patents": {"v":3774768,"path":os.path.join(input_dir, "patents.data")},
    "Reddit": {"v":232965,"path":os.path.join(input_dir, "reddit.data")}
}

knightKing = os.path.join(args.knightKing, 'build/bin')

knightKingWalks = {
    "Node2Vec": " -p 2.0 -q 0.5 -l 100 ", "PPR":" -t 0.001 ", "DeepWalk": " -l 100 ",
}

nextDoorApps = ["MVS", "PPR", "Node2Vec","DeepWalk","KHop","MultiRW","MVS","ClusterGCN","FastGCN", "LADIES"]
multiGPUApps = ["PPR", "Node2Vec","DeepWalk","KHop"]

results = {"KnightKing": {walk : {graph: -1 for graph in graphInfo} for walk in knightKingWalks},
           "SP": {walk : {graph: -1 for graph in graphInfo} for walk in nextDoorApps},
           "TP": {walk : {graph: -1 for graph in graphInfo} for walk in nextDoorApps},
           "LB": {walk : {graph: -1 for graph in graphInfo} for walk in nextDoorApps},
           "InversionTime": {walk : {graph: -1 for graph in graphInfo} for walk in nextDoorApps},
           "MultiGPU-LB": {walk : {graph: -1 for graph in graphInfo} for walk in nextDoorApps}}

def writeToLog(s):
    if not os.path.exists(logFile):
        open(logFile,"w").close()
    with open(logFile, "r+") as f:
        f.write(s)

writeToLog("=========Starting Run at %s=========="%(datetime.datetime.now().strftime("%d/%m/%Y %H:%M:%S")))

# for walk in knightKingWalks:
#     for graph in graphInfo:
#         times = []
#         for run in range(1):
#             walkBinary = os.path.join(knightKing, walk.lower()) + " -w %d "%graphInfo[graph]["v"] + \
#                 " -v %d"%graphInfo[graph]["v"] +\
#                 " -s weighted " + "-g " + graphInfo[graph]["path"] + \
#                  knightKingWalks[walk]   
#             print("Executing " + walkBinary)
#             writeToLog("Executing "+walkBinary)
#             status, output = subprocess.getstatusoutput(walkBinary)
#             writeToLog(output)
#             t = float(re.findall(r'total time ([\d\.]+)s', output)[0])
#             times += [t]

#         avg = sum(times)/len(times)

#         results["KnightKing"][walk][graph] = avg

# os.chdir(args.nextdoor)

# for app in nextDoorApps:
#     times = []
#     appBinary = os.path.join("build/tests/singleGPU", app.lower())
#     print ("Running ", appBinary)
#     writeToLog("Executing "+appBinary)
#     status, output = subprocess.getstatusoutput(appBinary)
#     writeToLog(output)
#     for technique in results:
#         if technique == "KnightKing" or technique == "InversionTime" or technique == "MultiGPU-LB":
#             continue
#         for graph in graphInfo:
#             print (app, graph, technique)
#             o = re.findall(r'%s\.%s%s.+%s\.%s%s'%(app, graph, technique,app,graph,technique), output, re.DOTALL)
#             if (len(o) == 0):
#                 print("Error executing %s for %s with input %s"%(technique, app, graph))
#                 continue
#             out = o[0]
#             end2end = re.findall(r'End to end time ([\d\.]+) secs', out)
#             results[technique][app][graph] = float(end2end[0])
#             if (technique == "LB"):
#                 inversionTime = re.findall(r'InversionTime: ([\d\.]+)', out)
#                 loadbalancingTime = re.findall(r'LoadBalancingTime: ([\d\.]+)', out)
#                 t = float(inversionTime[0]) + float(loadbalancingTime[0])
#                 results["InversionTime"][app][graph] = t

# if len(args.gpus) > 1:
#     #MultiGPU Results
#     for app in multiGPUApps:
#         times = []
#         appBinary = "CUDA_DEVICES="+args.gpus + " " + os.path.join("build/tests/multiGPU", app.lower())
#         print ("Running ", appBinary)
#         writeToLog("Executing "+appBinary)
#         status, output = subprocess.getstatusoutput(appBinary)
#         writeToLog(output)
#         print (output)
#         technique = "LB"
#         for graph in graphInfo:
#             print (app, graph, technique)
#             o = re.findall(r'%s\.%s%s.+%s\.%s%s'%(app, graph, technique,app,graph,technique), output, re.DOTALL)
#             if (len(o) == 0):
#                 printf("Error executing %s for %s with input %s"%(technique, app, graph))
#                 continue
#             out = o[0]
#             end2end = re.findall(r'End to end time ([\d\.]+) secs', out)
#             results["MultiGPU-LB"][app][graph] = float(end2end[0])
# else:
#     print ("Not taking MultiGPU results because only one GPU mentioned in 'gpus': ", args.gpus)
    
#Speedup Over KnightKing
print ("\n\nFigure 7 (a): Speedup Over KnightKing")
row_format = "{:>20}" * 3
print (row_format.format("Random Walk", "Graph", "Speedup"))
for walk in knightKingWalks:
    for graph in graphInfo:
        speedup = results["KnightKing"][walk][graph]/results["LB"][walk][graph]
        print (row_format.format(walk, graph, speedup))

#Speedup Over GNNs
import json
try:
    with open('gnnSamplingResults.json', 'r') as fp:
        samplingTimeResults = json.load(fp)
    print ("\n\nFigure 7 (b): Speedup Over Existing GNNs")
    row_format = "{:>30}" * 3
    print (row_format.format("Sampling App", "Graph", "Speedup"))
    for walk in nextDoorApps:
        if walk in knightKingWalks:
            continue
        for graph in graphInfo:
            gnnWalk = walk.lower() if walk != "MultiRW" else "graphsaint"
            if not gnnWalk in samplingTimeResults or not graph in samplingTimeResults[gnnWalk]:
                continue
            if (samplingTimeResults[gnnWalk][graph] < 0):
                speedup = "OOM"
            else:
                speedup = samplingTimeResults[gnnWalk][graph]/results["LB"][walk][graph]
            print (row_format.format(walk, graph, speedup))
except Exception as ee:
    print ("gnnSamplingResults.json did not exist. Did you execute runGNNSampling.py before this file?")

#Speedup Over SP and TP
print ("\n\nFigure 7 (c): Speedup Over SP and TP")
row_format = "{:>30}" * 4
print (row_format.format("Sampling App", "Graph", "Speedup over SP", "Speedup over TP"))
for walk in nextDoorApps:
    for graph in graphInfo:
        if graph == "Reddit":
            continue
        speedupSP = results["SP"][walk][graph]/results["LB"][walk][graph]
        speedupTP = results["TP"][walk][graph]/results["LB"][walk][graph]
        print (row_format.format(walk, graph, speedupSP, speedupTP))


print ("\n\nFigure 6: %age of Time Spent in Building scheduling index")
row_format = "{:>30}" * 3
print (row_format.format("Sampling App", "Graph", "%age of Time in Index"))
for walk in nextDoorApps:
    for graph in graphInfo:
        if graph == "Reddit":
            continue
        t = results["InversionTime"][walk][graph]/results["LB"][walk][graph]
        print (row_format.format(walk, graph, t * 100))

import json
with open('benchmarkResults.json', 'w') as fp:
    json.dump(results, fp)

# #Multi GPU results
# print ("\n\nFigure 10: Speedup of sampling using Multiple GPUs over 1 GPU")
# row_format = "{:>30}" * 3
# print (row_format.format("Sampling App", "Graph", "%age of Time in Index"))
# for walk in multiGPUApps:
#     for graph in graphInfo:
#         speedup = results["LB"][walk][graph]/results["MultiGPU-LB"][walk][graph]
#         print (row_format.format(walk, graph, speedup * 100))
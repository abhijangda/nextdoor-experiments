import argparse
import os
import subprocess
import re
import datetime 

logFile = os.path.join(os.getcwd(), "benchmarking.log")

parser = argparse.ArgumentParser(description='Benchmark')
parser.add_argument('-knightKing', type=str,
                    help='Path to KnightKing',required=True)
parser.add_argument('-nextdoor', type=str,
                    help='Path to NextDoor',required=True)
parser.add_argument('-printExistingResults', type=bool,
                    help='Print Results from previous invocation. Each invocation store results in benchmarkResults.json.',required=False)
# parser.add_argument('-runs', type=int, help="Number of Runs",required=True)
parser.add_argument('-gpus', type=str, help="CUDA DEVICES",required=False)

args = parser.parse_args()
args.nextdoor = os.path.abspath(args.nextdoor)
args.knightKing = os.path.abspath(args.knightKing)
printExistingResults = False if args.printExistingResults == None else args.printExistingResults
cwd = os.getcwd()

input_dir = os.path.join(args.nextdoor, "input")

#Run KnightKing Benchmarks
graphInfo = {
    "PPI": {"v": 56944, "path": os.path.join(input_dir, "ppi.data"), "w" : 5694400},
    "LiveJournal": {"v": 4576926, "path": os.path.join(input_dir, "LJ1.data"), "w": 4576926},
    "Orkut": {"v":3072441,"path":os.path.join(input_dir, "orkut.data"), "w":3072441},
    "Patents": {"v":3774768,"path":os.path.join(input_dir, "patents.data"), "w": 3774768},
    "Reddit": {"v":232965,"path":os.path.join(input_dir, "reddit.data"), "w": 2329650}
}

knightKing = os.path.join(args.knightKing, 'build/bin')

knightKingWalks = {
    "Node2Vec": " -p 2.0 -q 0.5 -l 100 ", "PPR":" -t 0.001 ", "DeepWalk": " -l 100 ",
}

nextDoorApps = ["MVS", "PPR", "Node2Vec","DeepWalk","KHop","Layer","MultiRW","ClusterGCN","FastGCN", "LADIES"]
multiGPUApps = ["PPR", "Node2Vec","DeepWalk","KHop","Layer"]

results = {"KnightKing": {walk : {graph: -1 for graph in graphInfo} for walk in knightKingWalks},
           "SP": {walk : {graph: -1 for graph in graphInfo} for walk in nextDoorApps},
           "TP": {walk : {graph: -1 for graph in graphInfo} for walk in nextDoorApps},
           "LB": {walk : {graph: -1 for graph in graphInfo} for walk in nextDoorApps},
           "InversionTime": {walk : {graph: -1 for graph in graphInfo} for walk in nextDoorApps},
           "MultiGPU-LB": {walk : {graph: -1 for graph in graphInfo} for walk in nextDoorApps}}

def writeToLog(s):
    if not os.path.exists(logFile):
        open(logFile,"w").close()
    with open(logFile, "a") as f:
        f.write(s)

writeToLog("=========Starting Run at %s=========="%(datetime.datetime.now().strftime("%d/%m/%Y %H:%M:%S")))

if not printExistingResults:
    if not os.path.exists(knightKing):
        print ("KnightKing directory '%s' does not exist. Make sure you have clone KnightKing and build all its apps."%(knightKing))

    if os.path.exists(knightKing):
        for walk in knightKingWalks:
            for graph in graphInfo:
                times = []
                for run in range(1):
                    walkBinary = os.path.join(knightKing, walk.lower()) + " -w %d "%graphInfo[graph]["w"] + \
                        " -v %d"%graphInfo[graph]["v"] +\
                        " -s weighted " + "-g " + graphInfo[graph]["path"] + \
                        knightKingWalks[walk]   
                    print("Executing " + walkBinary)
                    writeToLog("Executing "+walkBinary)
                    status, output = subprocess.getstatusoutput(walkBinary)
                    writeToLog(output)
                    t = float(re.findall(r'total time ([\d\.]+)s', output)[0])
                    times += [t]

                avg = sum(times)/len(times)

                results["KnightKing"][walk][graph] = avg

    os.chdir(args.nextdoor)

    for app in nextDoorApps:
        times = []
        appBinary = os.path.join("build/tests/singleGPU", app.lower())
        print ("Running ", appBinary)
        writeToLog("Executing "+appBinary)
        status, output = subprocess.getstatusoutput(appBinary)
        writeToLog(output)
        for technique in results:
            if technique == "KnightKing" or technique == "InversionTime" or technique == "MultiGPU-LB":
                continue
            for graph in graphInfo:
                print (app, graph, technique)
                o = re.findall(r'%s\.%s%s.+%s\.%s%s'%(app, graph, technique,app,graph,technique), output, re.DOTALL)
                if (len(o) == 0):
                    print("Error executing %s for %s with input %s"%(technique, app, graph))
                    continue
                out = o[0]
                end2end = re.findall(r'End to end time ([\d\.]+) secs', out)
                results[technique][app][graph] = float(end2end[0])
                if (technique == "LB"):
                    inversionTime = re.findall(r'InversionTime: ([\d\.]+)', out)
                    loadbalancingTime = re.findall(r'LoadBalancingTime: ([\d\.]+)', out)
                    t = float(inversionTime[0]) + float(loadbalancingTime[0])
                    results["InversionTime"][app][graph] = t

    if args.gpus != None and len(args.gpus) > 1:
        #MultiGPU Results
        for app in multiGPUApps:
            times = []
            appBinary = "CUDA_DEVICES="+args.gpus + " " + os.path.join("build/tests/multiGPU", app.lower())
            print ("Running ", appBinary)
            writeToLog("Executing "+appBinary)
            status, output = subprocess.getstatusoutput(appBinary)
            writeToLog(output)
            technique = "LB"
            for graph in graphInfo:
                print(app, graph, technique)
                o = re.findall(r'%s\.%s%s.+%s\.%s%s'%(app, graph, technique,app,graph,technique), output, re.DOTALL)
                if (len(o) == 0):
                    print("Error executing %s for %s with input %s"%(technique, app, graph))
                    continue
                out = o[0]
                end2end = re.findall(r'End to end time ([\d\.]+) secs', out)
                results["MultiGPU-LB"][app][graph] = float(end2end[0])
    else:
        print ("Not taking MultiGPU results because only one GPU mentioned in 'gpus': ", args.gpus)
else:
    os.chdir(cwd)
    if not os.path.exists('benchmarkResults.json'):
        print ('benchmarkResults.json does not exist. You should invoke the script without this flag to take results.')
        sys.exit(0)

    import json
    with open('benchmarkResults.json', 'r') as fp:
        results = json.load(fp)

os.chdir(cwd)
#Speedup Over KnightKing
print ("\n\nFigure 7 (a): Speedup Over KnightKing")
row_format = "{:>20}" * 3
print (row_format.format("Random Walk", "Graph", "Speedup"))
for walk in knightKingWalks:
    for graph in graphInfo:
        speedup = results["KnightKing"][walk][graph]/results["LB"][walk][graph]
        print (row_format.format(walk, graph, speedup))

import json
with open('benchmarkResults.json', 'w') as fp:
    json.dump(results, fp)

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
            if walk == "MultiRW":
                gnnWalk = "graphsaint"
            elif walk == "KHop":
                gnnWalk = "graphsage"
            else:
                gnnWalk = walk.lower()
                
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

# #Multi GPU results
if args.gpus is not None and len(args.gpus) > 1:
    if (results["MultiGPU-LB"][multiGPUApps[0]]["PPI"] == -1):
        print ("MultiGPU results were not obtained in the execution.")
    else:
        print ("\n\nFigure 10: Speedup of sampling using Multiple GPUs over 1 GPU")
        row_format = "{:>30}" * 3
        print (row_format.format("Sampling App", "Graph", "Speedup"))
        for walk in multiGPUApps:
            for graph in graphInfo:
                speedup = results["LB"][walk][graph]/results["MultiGPU-LB"][walk][graph]
                print (row_format.format(walk, graph, speedup))

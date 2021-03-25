import argparse
import os
import subprocess
import re
import datetime,sys

logFile = os.path.join(os.getcwd(), "perfanalysis.log")

parser = argparse.ArgumentParser(description='Benchmark')
parser.add_argument('-nextdoor', type=str,
                    help='Path to NextDoor',required=True)
parser.add_argument('-nvprof', type=str, help='Path to nvprof', required = True)
parser.add_argument('-metric',type=str,help='Metric to profile. Should be one of these: l2_read_transactions, warp_execution_efficiency, sm_efficiency, gst_efficiency', required=True)
# parser.add_argument('-runs', type=int, help="Number of Runs",required=True)

allMetrics = ["l2_read_transactions", "warp_execution_efficiency", "sm_efficiency", "gst_efficiency"]

args = parser.parse_args()
if args.metric not in allMetrics:
    print("Invalid metric '%s'. Metric must be one of '%s'."%(args.metric, str(allMetrics)))
    sys.exit(1)

args.nextdoor = os.path.abspath(args.nextdoor)
nvprofCommand = args.nvprof
cwd = os.getcwd()

input_dir = os.path.join(args.nextdoor, "input")

#Run KnightKing Benchmarks
graphInfo = {
    "PPI": {"v": 56944, "path": os.path.join(input_dir, "ppi.data"), "w" : 5694400},
    "LiveJournal": {"v": 4576926, "path": os.path.join(input_dir, "LJ1.data"), "w": 4576926},
    "Orkut": {"v":3072441,"path":os.path.join(input_dir, "orkut.data"), "w":3072441},
    "Patents": {"v":3774768,"path":os.path.join(input_dir, "patents.data"), "w": 3774768},
 #   # "Reddit": {"v":232965,"path":os.path.join(input_dir, "reddit.data"), "w": 2329650}
}

nextDoorApps = ["PPR", "Node2Vec", "DeepWalk", "KHop", "Layer"]
valuesDict = {"SP": {walk : {graph: -1 for graph in graphInfo} for walk in nextDoorApps},
                                             "LB": {walk : {graph: -1 for graph in graphInfo} for walk in nextDoorApps}
                                             }
L2CacheReads = {"metric":"l2_read_transactions", "values": dict(valuesDict)}
WarpExecutionEfficiency = {"metric":"warp_execution_efficiency", "values": dict(valuesDict)}
MultiProcessorActivity = {"metric":"sm_efficiency", "values": dict(valuesDict)}
GlobalStoreEfficiency = {"metric":"gst_efficiency", "values": dict(valuesDict)}

results = {"L2CacheReads": L2CacheReads,"WarpExecutionEfficiency": WarpExecutionEfficiency, 
           "MultiProcessorActivity":MultiProcessorActivity,"GlobalStoreEfficiency":GlobalStoreEfficiency}
nvprofCommand = nvprofCommand

techniqueCommand = {"SP": "SampleParallel", "LB" : "TransitParallel -l"}
LD_LIBRARY_PATH = os.getenv("LD_LIBRARY_PATH")

def writeToLog(s):
    if not os.path.exists(logFile):
        open(logFile,"w").close()
    with open(logFile, "a") as f:
        f.write(s)

writeToLog("=========Starting Run at %s=========="%(datetime.datetime.now().strftime("%d/%m/%Y %H:%M:%S")))

for app in nextDoorApps:
    appDir = app.lower()
    if app == "PPR" or app == "Node2Vec" or app == "DeepWalk":
        appDir = "randomwalks"
        if (args.metric == "gst_efficiency"):
            continue
    appDir = os.path.join(args.nextdoor, './src/apps/', appDir)
    print ("chdir %s"%appDir)
    os.chdir(appDir)
    c = "make bin -j"
    print ("Executing ", c)
    writeToLog("Executing "+ c)
    status, output = subprocess.getstatusoutput(c)
    if (status != 0):
        print ("Error executing '%s' in '%s'"%(c, os.getcwd()))
        writeToLog(output)
        continue

    for graph in graphInfo:
        for technique in techniqueCommand:
            kernels = ""
            if args.metric in ["warp_execution_efficiency", "sm_efficiency", "gst_efficiency"]:
                kernels = '"gridKernel|threadBlockKernel|identityKernel|samplingKernel|sampleParallelKernel"'
            
            templateCommand = 'sudo LD_LIBRARY_PATH=%s '%(LD_LIBRARY_PATH) + nvprofCommand + (' ' if kernels == "" else " --kernels " + kernels) + " --metrics " + args.metric + ' ./' + app + "Sampling -g %s -t edge-list -f binary -n 1 -k %s "
        
            if ((args.metric == "sm_efficiency" or args.metric == "gst_efficiency") and technique == "SP"):
                continue
            command = templateCommand%(graphInfo[graph]["path"], techniqueCommand[technique])
            print("Executing '%s'"%command)
            writeToLog("Executing '%s'"%command)
            status,output = subprocess.getstatusoutput(command)
            # print(output)
            writeToLog(output)
            if (status != 0):
                print("Error executing command")
                continue
            kernels = []
            for o in output[output.find('Kernel:'):].split("Kernel:"):
                if o.strip() == '':
                    continue
                if 'memset_kernel' not in o and 'init_curand' not in o:
                    kernels += [o]
            
            for result in results:
                if (results[result]["metric"] != args.metric):
                    continue
                totalValue = 0
                for kernel in kernels:
                    # print (kernel)
                    regexp = r"(\d+)\s+%s.+?([\d\.]+).+?([\d\.]+).+?([\d\.]+)"%(results[result]["metric"])
                    values = re.findall(regexp,kernel)
                    for value in values:
                        if result == "L2CacheReads":
                            totalValue += int(value[0]) * int(value[3])
                        else:
                            totalValue = max(totalValue, float(value[3]) if args.metric != "gst_efficiency" else (float(value[3])/(25/32.)*100.0))
                # print(totalValue)
                results[result]["values"][technique][app][graph] = totalValue


print(results)

if args.metric == "l2_read_transactions":
    print ("\n\nFigure 8 (a): L2 Cache Transactions")
    row_format = "{:>20}" * 3
    print (row_format.format("Application", "Graph", "Relative Value"))
    for app in nextDoorApps:
        for graph in graphInfo:
            relValue = results["L2CacheReads"]["values"]["LB"][app][graph]/results["L2CacheReads"]["values"]["SP"][app][graph]
            print (row_format.format(app, graph, relValue))
elif args.metric == "warp_execution_efficiency":
    print ("\n\nFigure 8 (b): Warp Execution Efficiency")
    row_format = "{:>20}" * 3
    print (row_format.format("Application", "Graph", "Relative Value"))
    for app in nextDoorApps:
        for graph in graphInfo:
            relValue = results["WarpExecutionEfficiency"]["values"]["LB"][app][graph]/results["WarpExecutionEfficiency"]["values"]["SP"][app][graph]
            print (row_format.format(app, graph, 1/relValue*2))
elif args.metric == "sm_efficiency":
    print ("\n\nTable 4: Multiprocessor Activity")
    row_format = "{:>20}" * 3
    print (row_format.format("Application", "Graph", "Value"))
    for app in nextDoorApps:
        for graph in graphInfo:
            relValue = results["MultiProcessorActivity"]["values"]["LB"][app][graph]
            print (row_format.format(app, graph, relValue))
elif args.metric == "gst_efficiency":
    print ("\n\nTable 4: GlobalStoreEfficiency")
    row_format = "{:>20}" * 3
    print (row_format.format("Application", "Graph", "Value"))
    for app in nextDoorApps:
        if app == "PPR" or app == "Node2Vec" or app == "DeepWalk":
            continue
        for graph in graphInfo:
            relValue = results["GlobalStoreEfficiency"]["values"]["LB"][app][graph]
            print (row_format.format(app, graph, relValue))

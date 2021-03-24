import argparse
import os
import subprocess
import re
import datetime 

logFile = os.path.join(os.getcwd(), "perfanalysis.log")

parser = argparse.ArgumentParser(description='Benchmark')
parser.add_argument('-nextdoor', type=str,
                    help='Path to NextDoor',required=True)
parser.add_argument('-nvprof', type=str, help='Path to nvprof', required = True)
# parser.add_argument('-runs', type=int, help="Number of Runs",required=True)

args = parser.parse_args()
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

L2CacheReads = {"metric":"l2_read_transactions", "values":{"SP": {walk : {graph: -1 for graph in graphInfo} for walk in nextDoorApps},
                                             "LB": {walk : {graph: -1 for graph in graphInfo} for walk in nextDoorApps}
                                             }
                                             }

results = {"L2CacheReads": L2CacheReads}
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
    appDir = os.path.join(args.nextdoor, './src/apps/', appDir)
    print ("Chdir %s"%appDir)
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
        templateCommand = 'sudo LD_LIBRARY_PATH=%s '%(LD_LIBRARY_PATH) + nvprofCommand + ' --metrics l2_read_transactions ' + './' + app + "Sampling -g %s -t edge-list -f binary -n 1 -k %s "
        for technique in techniqueCommand:
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
                totalValue = 0
                for kernel in kernels:
                    regexp = r"(\d+)\s+%s.+?(\d+)\s*(\d+)\s*(\d+)"%(results[result]["metric"])
                    values = re.findall(regexp,kernel)
                    for value in values:
                        totalValue += int(value[0]) * int(value[3])
                
                # print(totalValue)
                results[result]["values"][technique][app][graph] = totalValue


print(results)

#Speedup Over KnightKing
print ("\n\nFigure 8 (a): L2 Cache Transactions")
row_format = "{:>20}" * 3
print (row_format.format("Application", "Graph", "Relative Value"))
for app in nextDoorApps:
    for graph in graphInfo:
        relValue = results["L2CacheReads"]["values"]["LB"][app][graph]/results["L2CacheReads"]["values"]["SP"][app][graph]
        print (row_format.format(app, graph, relValue))
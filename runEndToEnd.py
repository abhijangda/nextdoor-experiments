import argparse, os, subprocess
import shutil, re
import datetime 

logFile = os.path.join(os.getcwd(), "end2endBenchmarking.log")

parser = argparse.ArgumentParser(description='Benchmark')
parser.add_argument('-nextdoor', type=str,
                    help='Path to NextDoor',required=True)

args = parser.parse_args()

cwd = os.getcwd()
input_dir = cwd
graphInfo = {
    "PPI": {"v": 56944, "path": os.path.join(input_dir, "ppi.data")},
    # "LiveJournal": {"v": 4847569, "path": os.path.join(input_dir, "LJ1.data")},
    # "Orkut": {"v":3072441,"path":os.path.join(input_dir, "orkut.data")},
    # "Patents": {"v":6009555,"path":os.path.join(input_dir, "patents.data")},
    "Reddit": {"v":232965,"path":os.path.join(input_dir, "reddit.data")}
}

def writeToLog(s):
    if not os.path.exists(logFile):
        open(logFile,"w").close()
    
    f = open(logFile, "r+")
    f.write(s)
    f.close()

writeToLog("=========Starting Run at %s=========="%(datetime.datetime.now().strftime("%d/%m/%Y %H:%M:%S")))

gnns = ['FastGCN', 'LADIES']
#Build sampling application in NextDoor folder
for gnn in gnns:
  d = os.path.join(args.nextdoor,'src/apps/',gnn.lower())
  os.chdir(d)
  writeToLog("Chdir to "+ d)
  writeToLog("Executing make")
  status, output = subprocess.getstatusoutput("make clean")
  writeToLog(output)
  status, output = subprocess.getstatusoutput("make -j")
  if (status != 0):
    print(output)
  writeToLog(output)
  #Copy libraries from NextDoor folder to GNN folder
  src = os.path.join(d, gnn+"SamplingPy3.so")
  dst = os.path.join(cwd, 'LADIES/',gnn+"SamplingPy3.so")
  shutil.copyfile(src, dst)

os.chdir(cwd)
baselineResults = {gnn.lower(): {graph: -1 for graph in graphInfo} for gnn in gnns}
nextdoorResults = {gnn.lower(): {graph: -1 for graph in graphInfo} for gnn in gnns}

def runForGNN(gnn):
  global results
  
  if (gnn == 'FastGCN' or gnn == 'LADIES'):
    os.chdir('./LADIES')
    gnnCommand = "python3 pytorch_ladies.py --cuda 0 --dataset %s --sample_method fastgcn --epoch_num 10 --n_iters 2"
  
  if gnn == 'graphsage':
    os.chdir('./GraphSAGE')
    gnnCommand = "python3 experiment/nextdoor_end2end.py %s"
  writeToLog("doing perf eval of %s"%gnn)
  status,output = subprocess.getstatusoutput("env -i bash -c 'source venv/bin/activate && env'")
  writeToLog(output)
  for line in output.split('\n'):
    (key, _, value) = line.partition("=")
    os.environ[key] = value
    
  for graph in graphInfo:
    c = gnnCommand % graph.lower()
    print(c)
    writeToLog("executing " + c)
    status,output = subprocess.getstatusoutput(c)
    writeToLog(output)
    samplerTimes = re.findall('end_to_end_time.+', output)
    for samplerTime in samplerTimes:
      s = re.findall(r'\((\w+)\)\s*([\.\d]+)', samplerTime)[0]
      samplerName = s[0]
      time = s[1]
      if 'nextdoor_' in samplerName:
        nextdoorResults[samplerName[len("nextdoor_"):]][graph] = float(time)
      else:
        baselineResults[samplerName][graph] = float(time)

runForGNN('graphsage')
 
#Print results
print ("\n\nTable 5: End-to-end speedups after integrating NextDoor in GNNs over vanilla GNNs")
row_format = "{:>30}" * 3
print (row_format.format("GNN", "Graph", "Speedup"))
for samplerName in baselineResults:
    for graph in graphInfo:
        speedup = baselineResults[samplerName][graph]/nextdoorResults[samplerName][graph]
        print (row_format.format(samplerName, graph, speedup))

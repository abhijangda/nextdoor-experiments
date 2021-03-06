import argparse, os, subprocess
import shutil, re
import datetime 

logFile = os.path.join(os.getcwd(), "samplingBenchmarking.log")

parser = argparse.ArgumentParser(description='Benchmark')
parser.add_argument('-nextdoor', type=str,
                    help='Path to NextDoor',required=True)
parser.add_argument('-graph_dir', type=str, help='Path to Graph Binary Dir', required=True)

args = parser.parse_args()
args.nextdoor = os.path.abspath(args.nextdoor)
args.graph_dir = os.path.abspath(args.graph_dir)

cwd = os.getcwd()
input_dir = cwd
graphInfo = {
    "PPI": {"v": 56944, "path": os.path.join(input_dir, "ppi.data")},
    # "LiveJournal": {"v": 4847569, "path": os.path.join(input_dir, "LJ1.data")},
    # "Orkut": {"v":3072441,"path":os.path.join(input_dir, "orkut.data")},
    # "Patents": {"v":6009555,"path":os.path.join(input_dir, "patents.data")},
    # "Reddit": {"v":232965,"path":os.path.join(input_dir, "reddit.data")}
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

samplingTimeResults = {gnn.lower(): {graph: -1 for graph in graphInfo} for gnn in gnns}
# nextdoorResults = {gnn.lower(): {graph: -1 for graph in graphInfo} for gnn in gnns}

def runForGNN(gnn):
  global results
  
  if (gnn == 'FastGCN' or gnn == 'LADIES'):
    os.chdir('./LADIES')
    gnnCommand = "python3 pytorch_ladies.py --cuda 0 --dataset %s --epoch_num 10 --n_iters 2 --graph_dir %s"
  writeToLog("doing perf eval of %s"%gnn)
  status,output = subprocess.getstatusoutput("env -i bash -c 'source venv/bin/activate && env'")
  writeToLog(output)
  for line in output.split('\n'):
    (key, _, value) = line.partition("=")
    os.environ[key] = value
    
  for graph in graphInfo:
    c = gnnCommand % (graph.lower(), args.graph_dir)
    print(c)
    writeToLog("executing " + c)
    status,output = subprocess.getstatusoutput(c)
    writeToLog(output)
    print(output)
    samplerTimes = re.findall('sampling_time.+', output)
    for samplerTime in samplerTimes:
      s = re.findall(r'\((\w+)\)\s*([\.\d]+)', samplerTime)
      samplerName = s[0][0]
      if('nextdoor_' in samplerName ):
        continue
      time = s[0][1]
      samplingTimeResults[samplerName][graph] = float(time)

runForGNN('FastGCN')
 
#Print results
print (samplingTimeResults)
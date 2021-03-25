import argparse, os, subprocess
import shutil, re
import datetime

logFile = os.path.join(os.getcwd(), "gnnSamplingBenchmarking.log")

parser = argparse.ArgumentParser(description='Benchmark')
parser.add_argument('-nextdoor', type=str,
                    help='Path to NextDoor',required=True)
parser.add_argument('-useSmallGraphs', type=bool, help='Use only PPI and Reddit',required=False)

args = parser.parse_args()
args.nextdoor = os.path.abspath(args.nextdoor)
if not hasattr(args,'useSmallGraphs'):
  args.useSmallGraphs = False

cwd = os.getcwd()
args.nextdoor = os.path.abspath(args.nextdoor)
input_dir = args.nextdoor
graph_dir = os.path.join(input_dir, "input")
graphInfo = {
   "PPI": {"v": 56944, "path": os.path.join(graph_dir, "ppi.data")},
  "LiveJournal": {"v": 4847569, "path": os.path.join(input_dir, "LJ1.data")},
    "Orkut": {"v":3072441,"path":os.path.join(input_dir, "orkut.data")},
  "Patents": {"v":6009555,"path":os.path.join(input_dir, "patents.data")},
   "Reddit": {"v":232965,"path":os.path.join(input_dir, "reddit.data")}
}

def writeToLog(s):
    if not os.path.exists(logFile):
        open(logFile,"w").close()

    f = open(logFile, "a")
    f.write(s)
    f.close()

writeToLog("=========Starting Run at %s=========="%(datetime.datetime.now().strftime("%d/%m/%Y %H:%M:%S")))


gnns = ['GraphSAGE', 'FastGCN', 'LADIES']
#Build sampling application in NextDoor folder
for gnn in gnns:
  appName = None
  if (gnn == 'FastGCN' or gnn == 'LADIES'):
    appName = gnn.lower()
  elif (gnn == 'GraphSAGE'):
    appName = 'khop'

  d = os.path.join(args.nextdoor,'src/apps/',appName)
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
  src = os.path.join(d, (gnn if(gnn != 'GraphSAGE') else "KHop") +"SamplingPy3.so")
  gnnDir = 'LADIES/' if gnn == 'FastGCN' else gnn
  dst = os.path.join(cwd, gnnDir, (gnn if(gnn != 'GraphSAGE') else "KHop")+ "SamplingPy3.so")
  shutil.copyfile(src, dst)


os.chdir(cwd)

gnns = ['FastGCN', 'LADIES','mvs','graphsaint','clustergcn','graphsage']
#Build sampling application in NextDoor folder

samplingTimeResults = {gnn.lower(): {graph: -1 for graph in graphInfo} for gnn in gnns}
# nextdoorResults = {gnn.lower(): {graph: -1 for graph in graphInfo} for gnn in gnns}

def runForGNN(gnn):
  global results
  gnnCommand = None
  if (gnn == 'FastGCN' or gnn == 'LADIES'):
    os.chdir('./LADIES')
    gnnCommand = "python3 pytorch_ladies.py --cuda 0 --dataset %s --epoch_num 10 --n_iters 2 --graph_dir %s"
  if gnn == 'mvs':
    os.chdir('./mvs_gcn')
    gnnCommand = "python main_experiments.py --dataset %s --graph_dir %s --batch_size 256 --samp_num 256 --cuda 0 --is_ratio 1.0 --batch_num 20 --n_stops 1000 --show_grad_norm 1 --n_layers 2"
  if gnn == 'graphsaint':
    os.chdir('./GraphSAINT')
    gnnCommand = "python -m graphsaint.tensorflow_version.train --dataset  %s --graph_dir %s --train_config train_config/mrw.yml --gpu 0 "
  if gnn == 'clustergcn':
    os.chdir('./cluster_gcn')
    gnnCommand = "python train.py --dataset %s  --graph_dir %s  --nomultilabel --num_layers 3 --num_clusters 1500 --bsize 20 --hidden1 32 --dropout 0.2 --weight_decay 0  --early_stopping 200 --num_clusters_val 20 --num_clusters_test 1 --epochs 1 --save_name $1  --learning_rate 0.005 --diag_lambda 0.0001 --novalidation"
  if gnn == 'graphsage':
    os.chdir('./GraphSAGE')
    #  python experiment/epoch_run_time.py dataset graphdir
    gnnCommand = "python experiment/epoch_run_time.py  %s %s"
  if gnnCommand == None :
    print("gnn name not found",gnn)
    return
  writeToLog("doing perf eval of %s"%gnn)
  status,output = subprocess.getstatusoutput("env -i bash -c 'source venv/bin/activate && env'")
  writeToLog(output)
  for line in output.split('\n'):
    (key, _, value) = line.partition("=")
    os.environ[key] = value

  for graph in graphInfo:
    if (args.useSmallGraphs and graph in ['Patents', 'Orkut', 'LiveJournal']):
      continue
    if graph == 'LiveJournal':
        if gnn.lower() == 'graphsage' or gnn.lower() == 'mvs' or gnn.lower() == 'clustergcn':
            samplingTimeResults[gnn.lower()][graph] = -1
            continue
    if graph == 'Orkut':
        samplingTimeResults[gnn.lower()][graph] = -1
        continue
    if gnn.lower() == 'graphsaint' and graph in ['Patents', 'Orkut', 'LiveJournal']:
      continue
    if gnn.lower() in ['fastgcn', 'ladies'] and graph == 'Orkut':
      continue
    c = gnnCommand % ('LJ1' if graph == 'LiveJournal' else graph.lower(), graph_dir)
    print(c)
    writeToLog("executing " + c)
    status,output = subprocess.getstatusoutput(c)
    if status != 0:
      print("Graph '%s' leads to OOM with GNN '%s'" %(graph,gnn))
      continue
    writeToLog(output)
    samplerTimes = re.findall('sampling_time.+', output)
    for samplerTime in samplerTimes:
      try:
        s = re.findall(r'\((\w+)\)\s*([\.\d]+)', samplerTime)
        samplerName = s[0][0]
        if 'nextdoor_' in samplerName:
          continue
        time = s[0][1]
        samplingTimeResults[samplerName][graph] = float(time)
      except Exception as e:
        writeToLog("Exception:" + str(s) + "\n{0}".format(e))
  os.chdir(cwd)

runForGNN('clustergcn')
runForGNN('graphsaint')
runForGNN('mvs')
runForGNN('FastGCN')
runForGNN('graphsage')

#Print results
import json
with open('gnnSamplingResults.json', 'w') as fp:
    json.dump(samplingTimeResults, fp)
print(samplingTimeResults)

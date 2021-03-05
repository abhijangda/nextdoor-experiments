import argparse, os, subprocess
import shutil, re

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
    # "Reddit": {"v":232965,"path":os.path.join(input_dir, "reddit.data")}
}

gnns = ['FastGCN', 'LADIES']
#Build sampling application in NextDoor folder
# for gnn in gnns:
#   d = os.path.join(args.nextdoor,'src/apps/',gnn.lower())
#   print(d)
#   os.chdir(d)
#   status, output = subprocess.getstatusoutput("make clean")
#   status, output = subprocess.getstatusoutput("make -j")
#   print (output)
#   #Copy libraries from NextDoor folder to GNN folder
#   src = os.path.join(d, gnn+"SamplingPy3.so")
#   dst = os.path.join(cwd, 'LADIES/',gnn+"SamplingPy3.so")
#   shutil.copyfile(src, dst)

os.chdir(cwd)
baselineResults = {gnn.lower(): {graph: -1 for graph in graphInfo} for gnn in gnns}
nextdoorResults = {gnn.lower(): {graph: -1 for graph in graphInfo} for gnn in gnns}

def runForGNN(gnn):
  global results
  if (gnn == 'FastGCN' or gnn == 'LADIES'):
    os.chdir('./LADIES')
    gnnCommand = "python3 pytorch_ladies.py --cuda 0 --dataset %s --sample_method fastgcn --epoch_num 10 --n_iters 2"

  status,output = subprocess.getstatusoutput("env -i bash -c 'source venv/bin/activate && env'")
  for line in output.split('\n'):
    (key, _, value) = line.partition("=")
    os.environ[key] = value
    
  for graph in graphInfo:
    c = gnnCommand % graph.lower()
    status,output = subprocess.getstatusoutput(c)
    samplerTimes = re.findall('end_to_end_time.+', output)
    for samplerTime in samplerTimes:
      s = re.findall(r'\((\w+)\)\s*([\.\d]+)', samplerTime)[0]
      samplerName = s[0]
      time = s[1]
      if 'nextdoor_' in samplerName:
        nextdoorResults[samplerName[len("nextdoor_"):]] = time
      else:
        baselineResults[samplerName] = time

runForGNN('FastGCN')
#!/usr/bin/python

import subprocess
import re
# file = (filelocation, noVertex, noWalkers) 
files = [("../datasets/weighted-cit-Patents.data",3774768,1),
         ("../datasets/soc-LiveJournal1.data",4847571,1),
         ("../datasets/com-orkut.data",3072441,1),
         ("../datasets/com-friendster.data",65608366,1)]

files_d = [("./build/karate.data",100,1), 
         ("./build/karate.data",100,1)]

cmds = {"node2vec":"./build/bin/node2vec -g {0} -v {1} -w {2} -s weighted -l 10 -p 2 -q 0.5 -o ./build/out/walks.txt",
        "deepwalk":"./build/bin/deepwalk -g {0} -v {1} -w {2} -s weighted -l 10  -o ./build/out/walks.txt",
        "ppr": "./build/bin/ppr -g {0} -v {1} -w {2} -s weighted -t .3  -o ./build/out/walks.txt",
        #"metapath":"./bin/metapath -g {0} -v {1} -w {2} -s weighted -l 10  -o ./out/walks.txt"
        }

parseTime = "| grep total\ time|  cut -d " " -f 3 "

def run():
    w = open('experiments.txt','a')
    w.write("filename   | walk | time \n" )
    for fname,nv,nw in files:
        for k in cmds.keys():
            print(cmds[k].format(fname,nv,nw))
            time = subprocess.check_output(cmds[k].format(fname,nv,nw),shell=True)         
            time = re.search("total time ([\d|\.]*)s",time).group(1)
            w.write("{} | {} | {} \n".format(fname, k, time ))
    w.close()

if __name__=="__main__":
    run()

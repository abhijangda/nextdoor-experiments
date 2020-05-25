#!/usr/bin/python

import subprocess
import re
# file = (filelocation, noVertex, noWalkers,unweighted file location)
files = [
("../datasets/weighted-cit-Patents.data", 6009556,6009556,"../datasets/unweighted-cit-Patents.data"),
("../datasets/soc-LiveJournal1.data", 4847593,4847593,"../datasets/unweighted-soc-LiveJournal1.data"),
("../datasets/com-orkut.data", 3072640 ,3072640,"../datasets/unweighted-com-orkut.ungraph.data"),
("../datasets/com-friendster.data",124836199 ,124836199,"../datasets/unweighted-com-friendster.ungraph.data"),
("../datasets/weighted_ppi_edgelist.data",56950,56950,"../datasets/unweighted-ppi_edgelist.data"),
("../datasets/weighted_reddit_edgelist.data ",232970,232970,"../datasets/unweighted-reddit_edgelist.data"),
         ]


files_d = [("./build/karate.data",100,1),
         ("./build/karate.data",100,1)]

cmds = {"node2vec":"./build/bin/node2vec -g {0} -v {1} -w {2} -s {3} -l 1  -p 2 -q 0.5 ",
        "deepwalk":"./build/bin/deepwalk -g {0} -v {1} -w {2} -s {3} -l 1",
        "ppr": "./build/bin/ppr -g {0} -v {1} -w {2} -s {3} -t .1 ",
        #"metapath":"./bin/metapath -g {0} -v {1} -w {2} -s weighted -l 10  -o ./out/walks.txt"
        }

parseTime = "| grep total\ time|  cut -d " " -f 3 "

WALKLENGTH=1
def run():
    w = open('experiments.txt','a')
    w.write("filename   | walk | time \n" )
    for fname,nv,nw,wfname  in files:
        for k in cmds.keys():
            print(cmds[k].format(fname,nv,nw,"weighted"))
            time = subprocess.check_output(cmds[k].format(fname,nv,nw,"weighted"),shell=True)
            time = re.search("total time ([\d|\.]*)s",time).group(1)
            w.write("{} | {} | {} \n".format(fname, k, time ))
            print(cmds[k].format(wfname,nv,nw,"unweighted"))
            time = subprocess.check_output(cmds[k].format(wfname,nv,nw,"unweighted"),shell=True)
            time = re.search("total time ([\d|\.]*)s",time).group(1)
            w.write("{} | {} | {} \n".format(wfname, k, time ))

    w.close()

if __name__=="__main__":
    run()

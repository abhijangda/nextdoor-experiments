import sys,os
from os import path
sys.path.insert(0, os.getcwd())
from graphsage.utils import load_data

if __name__ == "__main__" :
    import sys
    PREFIX = sys.argv[1]
    print("PREFIX {}".format(PREFIX))
    G, feats, id_map, walks, class_map = load_data(PREFIX)
    fp = open("edgelist",'w')
    for nodeid in G.nodes():
        id_1 = id_map[nodeid]
        for neighbor in G.neighbors(nodeid):
            id_2 = id_map[neighbor]
            fp.write("{}\t{}\n".format(id_1,id_2))
    fp.close()


# Goal is to covert a SNAP dataset file of

import networkx as nx
from networkx.readwrite import json_graph
import json, random
#  Generate 4 files.
# (1) -G.json file
#  Dictionary - keys
#     1. directed - False
#     2. graph - {'name':"disjoint union()'} - ignore this parameter
#     3. nodes - List of node
#                     Each node = dictionary {test,label,id,val,feature}
#     4. links
#     5. multigraph
# 2 -class_map.json
# dictionary {id:id}
# 3 -id_map.json.
# 4 -feats.npy

import numpy as np

# FEATURE_SIZE = 50
# NO_LABELS= 121
# Reddit
#     classes labels - integer (0,40)
#     feature size - 602 vector
# PPi
#     class label - 121 vector
#     feature  50 vector

FEATURE_SIZE = 2
NO_LABELS = 512

def generate(filename, prefix, folder):
    import os
    if not os.path.exists(folder):
        os.makedirs(folder)
    G = nx.Graph()
    nodes = []
    with open(filename) as fp:
        for line in fp:
            if line.startswith("#"):
                continue
            a,b = line.split()
            a,b = int(a),int(b)
            G.add_edge(a,b)
            nodes.extend([a,b])
    nodes = set(nodes)
    max_node = len(nodes)
    feature_npy = np.zeros((max_node+1,FEATURE_SIZE))
    id_map = {}
    class_map = {}
    idx = 0
    labels = range(NO_LABELS)
    for node in nodes:
        a = random.random()
        test = False
        val = False
        if a<.20:
            test = True
            val = False
            if a <0.1:
                test = False
                val = True
        # label = [random.choice([0,1]) for i in range(NO_LABELS)]
        label = random.choice(labels)
        feature = [random.choice([0.0,1.0]) for i in range(FEATURE_SIZE)]
        feature_npy[idx] = feature
        G.add_node(node, {"test": test, "label": label, "val": val})
        # G.add_node(node,{"test":test,"label":label,"val":val,"feature":feature})
        id_map[node] = idx
        class_map[node] = label
    data = json_graph.node_link_data(G)
    with open(folder+"/"+prefix+"-G.json",'w') as f:
        json.dump(data, f )
    with open(folder + "/" + prefix + "-id_map.json",'w' ) as f:
        json.dump(id_map, f)
    with open(folder + "/" + prefix + "-class_map.json",'w' ) as f:
        json.dump(class_map, f)
    with open(folder + "/" + prefix + "-feats.npy",'w') as f:
        np.save(f,feature_npy)


#  Usage:
#     python dataset_generation.py <filename> <prefix> <folder_location>
# python experiment/dataset_generation.py example_data/com-amazon.ungraph.txt amazon  example_data/amazon

if __name__ == "__main__":
    import sys
    if len(sys.argv) != 4:
        print("Incorrectly used")
        exit(1)
    filename = sys.argv[1]
    prefix = sys.argv[2]
    folder_location = sys.argv[3]
    generate(filename,prefix,folder_location)
    print("successfully done! {} {} {}".format(filename, prefix, folder_location))
import sys,os
from os import path

import networkx as nx
import random
import tensorflow.compat.v1 as tf
import numpy as np
tf.disable_eager_execution()
tf.disable_v2_behavior()

sys.path.insert(0, os.getcwd())
import time
from graphsage.utils import load_data, run_random_walks
from graphsage.minibatch import NodeMinibatchIterator, EdgeMinibatchIterator, NodeMinibatchIteratorWithKHop
from graphsage.neigh_samplers import UniformNeighborSampler
from graphsage.models import  SAGEInfo
import tensorflow.compat.v1 as tf

PREFIX = "./example_data/toy-ppi"
file = None
N_WALKS = 50

# flags = tf.app.flags
# FLAGS = flags.FLAGS
# flags.DEFINE_integer('gpu', 0, "which gpu to use.")
# tf.app.flags.DEFINE_boolean('log_device_placement', False,
#                             """Whether to log device placement.""")
os.environ["CUDA_VISIBLE_DEVICES"]=str(0)

results = {}

from ctypes import *
from ctypes.util import *

libgraphPath = '../graph_loading/libgraph.so'
#l1 = 'libGraph.so'
# print(find_library(l1))
libgraph = CDLL(libgraphPath)
libgraph.loadgraph.argtypes = [c_char_p]


def custom_dataset(DATA):
    MAX_LABELS = 10
    MAX_FEATURE_SIZE = 8
    filename = DATA+".data"
    if not os.path.exists(filename):
        raise Exception("'%s' do not exist"%(filename))

    graphPath = bytes(filename, encoding='utf8')
    libgraph.loadgraph(graphPath)
    libgraph.getEdgePairList.restype = np.ctypeslib.ndpointer(dtype=c_int, shape=(libgraph.numberOfEdges(), 2))

    print("Graph Loaded in C++")

    edges = libgraph.getEdgePairList()
    # print("Number of Edges", d.numberOfEdges())
    # print("Number of Vertices", d.numberOfVertices())
    # print (edges)
    G = nx.Graph()
    print("Loading networkx graph")
    G.add_edges_from(edges)
    id_map = {}
    class_map = {}
    i = 0
    for n in G:
        id_map[n] = i
        i = i + 1
        class_map[n] = random.randint(0,MAX_LABELS)
        r = random.random()
        G.node[n]['test'] = False
        G.node[n]['val'] = False
        G.node[n]['train_removed'] = False
        if r >=.8 and r <.9:
            G.node[n]['val'] = True
        if r>= .9:
            G.node[n]['test']=True
    max_nodes = max(G.nodes()) + 1
    features = np.random.rand(max_nodes,MAX_FEATURE_SIZE)
    print ("features created")
    toret = (G, features, id_map, None, class_map)
    print("Returned tuple created")
    return toret



def add_to_dict(k,v):
    global results
    results[k] = v

def create_measurement_file():
    e = path.exists("experiments.txt")
    global results
    global file
    file = open("experiments.txt", 'a')
    if not e:
        file.write ("Dataset | BatchSize | GPU | Sup-Sampling | Nextdoor-Sampling | Sup-Epoch | Nextdoor-Epoch \n")# print header
        pass
    # print values from dictionary
    file.write("{} | {} | {} | {}| {} |{} | {} \n".format(results['DATASET'], results['BATCHSIZE'], results['GPU'],
                                     results['SSAMPLE'],results['NEXTSAMPLE'],
                                                     results['SEPOCH'],results['NEXTSEPOCH']))
    file.close()

def getMiniBatchIterator(G, id_map, class_map, num_classes,batch_size):
    placeholders = {
        'labels': tf.placeholder(tf.float32, shape=(None, num_classes), name='labels'),
        'batch': tf.placeholder(tf.int32, shape=(None,), name='batch1'),
        'dropout': tf.placeholder_with_default(0., shape=(), name='dropout'),
        'batch_size': tf.placeholder(tf.int32, name='batch_size'),
    }
    minibatch = NodeMinibatchIterator(G,
                                      id_map,
                                      placeholders,
                                      class_map,
                                      num_classes,
                                      batch_size=batch_size,
                                      max_degree=128)
    return minibatch

def getSampledBatchIterator(G, id_map, class_map, num_classes,batch_size):
    placeholders = {
        'labels': tf.placeholder(tf.float32, shape=(None, num_classes), name='labels'),
        'batch': tf.placeholder(tf.int32, shape=(None), name='batch1'),
        'dropout': tf.placeholder_with_default(0., shape=(), name='dropout'),
        'batch_size': tf.placeholder(tf.int32, name='batch_size'),
        'hop1': tf.placeholder(tf.int32, shape=(None), name='hop1'),
        'hop2': tf.placeholder(tf.int32, shape=(None), name='hop2'),
    }

    layer_infos_top_down = [SAGEInfo("node", None, 25,128),
                   SAGEInfo("node", None, 10, 128)]

    minibatch = NodeMinibatchIteratorWithKHop(G,
                                      id_map,
                                      placeholders,
                                      class_map,
                                      num_classes,
                                      layer_infos_top_down,
                                      batch_size=batch_size,
                                      max_degree=128,
                                      )
    return minibatch

def nextdoor_sampling(minibatch):
    sess = tf.Session()
    minibatch.shuffle()
    start_time = time.time()
    while not minibatch.end():
        feed_dict, _ = minibatch.next_minibatch_feed_dict()
        sess.run([minibatch.placeholders["hop2"],minibatch.placeholders["hop1"]], feed_dict)
    end_time = time.time()
    sess.close()
    add_to_dict("NEXTSAMPLE", (end_time - start_time))

def supervised_sampling(minibatch):
    pruned_adj_matrix = tf.constant(minibatch.adj, dtype=tf.int32)
    sampler = UniformNeighborSampler(pruned_adj_matrix)
    sample1 = sampler((minibatch.placeholders["batch"], 10))
    s_tv = tf.reshape(sample1, [tf.shape(sample1)[0] * 10,])
    sample2 = sampler((s_tv,25))
    sess = tf.Session()
    minibatch.shuffle()
    start_time = time.time()
    while not minibatch.end():
        feed_dict,_ = minibatch.next_minibatch_feed_dict()
        sess.run(sample2 , feed_dict)
    end_time = time.time()
    sess.close()
    add_to_dict("SSAMPLE",(end_time - start_time))


def supervised_epoch_time(G,feats, id_map, walks, class_map, batch_size):
    tf.reset_default_graph()
    flags = tf.app.flags
    FLAGS = flags.FLAGS
    del_all_flags(tf.flags.FLAGS)
    from graphsage.supervised_train import train
    FLAGS.train_prefix = PREFIX
    FLAGS.model = 'graphsage_mean'
    FLAGS.batch_size = batch_size
    FLAGS.sigmoid = True
    #train_data = load_data((G,id_map,class_map,num_classes,batch_size))
    time = train((G,feats,id_map,walks,class_map),batch_size)
    print("end_to_end_time(graphsage)", time)
    add_to_dict('SEPOCH',time)

def nextdoor_supervised_epoch_time(G,feats,id_map,walks,class_map, batch_size):
    tf.reset_default_graph()
    flags = tf.app.flags
    FLAGS = flags.FLAGS
    del_all_flags(tf.flags.FLAGS)
    from graphsage.sampled_supervised_train import  train
    FLAGS.train_prefix = PREFIX
    FLAGS.model = 'graphsage_mean'
    FLAGS.sigmoid = True
    FLAGS.batch_size = batch_size
    #train_data = load_data(FLAGS.train_prefix)
    time = train((G,feats,id_map,walks,class_map),batch_size)
    print("end_to_end_time(nextdoor_graphsage)", time)
    add_to_dict('NEXTSEPOCH', time)

def del_all_flags(FLAGS):
    flags_dict = FLAGS._flags()
    keys_list = [keys for keys in flags_dict]
    for keys in keys_list:
        FLAGS.__delattr__(keys)


def run():
    global PREFIX
    import sys
    DATA = sys.argv[1]
    #DATA = sys.argv[2]
    batchSize = 32

    add_to_dict("DATASET",(DATA))
    is_available = tf.test.is_gpu_available(
        cuda_only=False, min_cuda_compute_capability=None
    )
    add_to_dict("BATCHSIZE",batchSize)
    add_to_dict ("GPU",is_available)
    G, feats, id_map, walks, class_map = custom_dataset(DATA)
    print("Number of nodes {}".format(G.number_of_nodes()))
    print("Number of Edges {}".format(G.number_of_edges()))
    import sys
    if isinstance(list(class_map.values())[0], list):
        num_classes = len(list(class_map.values())[0])
    else:
        num_classes = len(set(class_map.values()))
    minibatch = getMiniBatchIterator(G, id_map, class_map, num_classes,batchSize)
    supervised_sampling(minibatch)
    minibatch = getSampledBatchIterator(G,id_map, class_map, num_classes,batchSize)
    nextdoor_sampling(minibatch)
    supervised_epoch_time(G,feats,id_map,walks,class_map,batchSize)
    nextdoor_supervised_epoch_time(G,feats,id_map,walks,class_map,batchSize)
    #create_measurement_file()
    print("All Done !!! ")


'''
    how to run !!!
    python experiment/epoch_run_time.py ./example_data/toy-ppi [batch_size]
'''
if __name__ == "__main__":
    # print("hello world")
    run()
    # run_single_experiment()

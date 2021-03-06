import networkx as nx
from ctypes import CDLL
import sys,os
from os import path
sys.path.insert(0, os.getcwd())
import time,random
#from graphsage.utils import load_data, run_random_walks
from graphsage.minibatch import NodeMinibatchIterator,EdgeMinibatchIterator
from graphsage.neigh_samplers import UniformNeighborSampler
import tensorflow.compat.v1 as tf
import numpy as np
tf.disable_eager_execution()
tf.disable_v2_behavior()
PREFIX = "./example_data/toy-ppi"
file = None
N_WALKS = 50

# flags = tf.app.flags
# FLAGS = flags.FLAGS
# flags.DEFINE_integer('gpu', 0, "which gpu to use.")
# tf.app.flags.DEFINE_boolean('log_device_placement', False,
#                             """Whether to log device placement.""")
os.environ["CUDA_VISIBLE_DEVICES"]=str(0)

from ctypes import *
from ctypes.util import *

libgraphPath = '../graph_loading/libgraph.so'
#l1 = 'libGraph.so'
# print(find_library(l1))
libgraph = CDLL(libgraphPath)
libgraph.loadgraph.argtypes = [c_char_p]


results = {}

def custom_dataset(graph_dir, dataset_str):
    MAX_LABELS = 10
    MAX_FEATURE_SIZE = 8
    filename = os.path.join(graph_dir, dataset_str+".data")
    if not os.path.exists(filename):
        raise Exception("Graph %s at '%s' do not exist"%(dataset_str, filename))

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

    
def load_data(graph_dir, dataset_str):
    if dataset_str == "ppi" or dataset_str == "reddit" or dataset_str == 'orkut' or dataset_str == "LJ1" or dataset_str == 'livejournal' or dataset_str == 'patents':
        return custom_dataset(graph_dir, dataset_str)


def add_to_dict(k,v):
    global results
    results[k] = v

def create_measurement_file():
    e = path.exists("experiments.txt")
    global results
    global file
    file = open("experiments.txt", 'a')
    if not e:
        file.write ("Dataset | GPU |  AdjMatrix | Random Wlk (CPU) | Sup-sampling  | Sup-Epoch | UnSup-sampling | UnSup-epoch \n")# print header
        pass
    # print values from dictionary
    file.write("{} | {} | {}| {} | {} | {} | {} | {} \n".format(results['DATASET'],results['GPU'], results['ADJMATRIX'],
                                                                results['RWK'],results['SSAMPLE']
                                                       , results['SEPOCH'],results['UNSSAMPLE'], results['UNSEPOCH']))
    file.close()





# # Measure Look up time
def time_to_do_random_walks(G):
    nodes = [n for n in G.nodes_iter() if G.node[n]["val"] or G.node[n]["test"]]
    start_time = time.time()
    run_random_walks(G, nodes, num_walks=N_WALKS)
    end_time = time.time()
    add_to_dict("RWK",end_time - start_time)

def time_for_unsupervised_sampling(G, id_map, walks, num_classes):
    tf.reset_default_graph()
    placeholders = {
        'labels': tf.placeholder(tf.float32, shape=(None, num_classes), name='labels'),
        'batch1': tf.placeholder(tf.int32, shape=(None,), name='batch1'),
        'batch2': tf.placeholder(tf.int32, shape=(None,), name='batch2'),
        'dropout': tf.placeholder_with_default(0., shape=(), name='dropout'),
        'batch_size': tf.placeholder(tf.int32, name='batch_size'),
    }
    minibatch = EdgeMinibatchIterator(G, id_map, placeholders, walks, batch_size=512, max_degree=100)
    label = tf.cast(minibatch.placeholders["batch2"], dtype=tf.int64)
    labels = tf.reshape(label,[placeholders['batch_size'],1])
    neg_samples, _, _ = (tf.nn.fixed_unigram_candidate_sampler(
        true_classes=labels,
        num_true=1,
        num_sampled=20,
        unique=False,
        range_max=len(minibatch.deg),
        distortion=0.75,
        unigrams=minibatch.deg.tolist()))
    pruned_adj_matrix = tf.constant(minibatch.adj, dtype=tf.int32)
    sampler = UniformNeighborSampler(pruned_adj_matrix)
    def get_two_hop_sampled(input_t):
        sample1 = sampler((input_t, 25))
        reshaped_sample1 = tf.reshape(sample1, [tf.shape(sample1)[0] * 25, ])
        sample2 = sampler((reshaped_sample1, 10))
        return sample2
    source = get_two_hop_sampled(minibatch.placeholders["batch1"])
    target = get_two_hop_sampled(minibatch.placeholders["batch2"])
    neg_samples = get_two_hop_sampled(neg_samples)
    minibatch.shuffle()
    sess = tf.Session()
    start_time = time.time()
    i = 0
    max_steps = 1000
    while not minibatch.end():
        feed_dict = minibatch.next_minibatch_feed_dict()
        sess.run([source,target,neg_samples], feed_dict)
        i = i + 1
        if(i>1000):
            break
    print("Total number of steps run {}".format(i))
    end_time = time.time()
    print("Sampling time with negative {}".format(end_time - start_time))
    sess.close()
    add_to_dict("UNSSAMPLE",(end_time - start_time))



def getMiniBatchIterator(G, id_map, class_map, num_classes):
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
                                      batch_size=512,
                                      max_degree=128)
    return minibatch

def time_to_do_adj_matrix_construction(minibatch):
    start_time = time.time()
    minibatch.construct_adj()
    end_time = time.time()
    add_to_dict("ADJMATRIX",(end_time - start_time))

def supervised_sampling(minibatch):
    pruned_adj_matrix = tf.constant(minibatch.adj, dtype=tf.int32)
    sampler = UniformNeighborSampler(pruned_adj_matrix)
    sample1 = sampler((minibatch.placeholders["batch"], 10))
    s_tv = tf.reshape(sample1, [tf.shape(sample1)[0] * 10,])
    sample2 = sampler((s_tv,28))
    sess = tf.Session()
    minibatch.shuffle()
    start_time = time.time()
    while not minibatch.end():
        feed_dict,_ = minibatch.next_minibatch_feed_dict()
        sess.run(sample2 , feed_dict)
    end_time = time.time()
    sess.close()
    print("sampling_time (graphsage)",(end_time - start_time))
    add_to_dict("SSAMPLE",(end_time - start_time))

def supervised_epoch_time(G,feats,id_map,walks,class_map):
    tf.reset_default_graph()
    flags = tf.app.flags
    FLAGS = flags.FLAGS
    from graphsage.supervised_train import train
    FLAGS.train_prefix = PREFIX
    FLAGS.model = 'graphsage_mean'
    FLAGS.sigmoid = True
    #train_data = load_data(FLAGS.train_prefix)
    time = train((G,feats,id_map,walks,class_map))
    print("training_time:",time)
    add_to_dict('SEPOCH',time)

def del_all_flags(FLAGS):
    flags_dict = FLAGS._flags()
    keys_list = [keys for keys in flags_dict]
    for keys in keys_list:
        FLAGS.__delattr__(keys)

def unsupervised_epoch_time(PREFIX):
    tf.reset_default_graph()
    flags = tf.app.flags
    FLAGS = flags.FLAGS
    del_all_flags(tf.flags.FLAGS)
    from graphsage.unsupervised_train import train
    FLAGS.model = 'gcn'
    FLAGS.validate_iter = 10
    FLAGS.max_total_steps = 1000
    train_data = load_data(FLAGS.train_prefix,load_walks=True)
    t = train(train_data)
    print("UNSEPOCH {}".format(t))
    add_to_dict("UNSEPOCH",t)

def run():
    global PREFIX
    import sys
    dataset_dir = sys.argv[2]
    file_name = sys.argv[1]
    #add_to_dict("DATASET",(PREFIX))
    is_available = tf.test.is_gpu_available(
        cuda_only=False, min_cuda_compute_capability=None
    )
    add_to_dict ("GPU",is_available)
    G, feats, id_map, walks, class_map = load_data(dataset_dir,file_name)
    #print("Data Loading Done")
    #print("Number of nodes {}".format(G.number_of_nodes()))
    #print("Number of Edges {}".format(G.number_of_edges()))
    #import sys
    if isinstance(list(class_map.values())[0], list):
        num_classes = len(list(class_map.values())[0])
    else:
        num_classes = len(set(class_map.values()))
    #time_to_do_random_walks(G)
    minibatch = getMiniBatchIterator(G, id_map, class_map, num_classes)
    #time_to_do_adj_matrix_construction(minibatch)
    supervised_sampling(minibatch)
    #time_for_unsupervised_sampling(G, id_map, walks , num_classes)
    supervised_epoch_time(G,feats,id_map,walks,class_map)
    #unsupervised_epoch_time(PREFIX)
    #create_measurement_file()
    #print("All Done !!! ")

def print_adj_matrix_to_file():
    global PREFIX
    import sys
    PREFIX = sys.argv[1]
    print("PREFIX {}".format(PREFIX))
    G, feats, id_map, walks, class_map = load_data(PREFIX)
    if isinstance(list(class_map.values())[0], list):
        num_classes = len(list(class_map.values())[0])
    else:
        num_classes = len(set(class_map.values()))
    minibatch = getMiniBatchIterator(G, id_map, class_map, num_classes)
    adj = minibatch.adj
    fp = open("_adj_matrix","w")
    # fp.write("DATASET" + PREFIX + "\n")
    # fp.write("nd1_id : < list of 128 sampled neighbours> \n")
    for ndid in G.nodes():
        fp.write("{} ".format(int(id_map[ndid])))
        neighbours = adj[id_map[ndid]]
        for n in neighbours:
            fp.write("{} ".format(int(n)))
        fp.write("\n")
    fp.close()


def run_single_experiment():
    print_adj_matrix_to_file()

'''
    how to run !!!
    python experiment/epoch_run_time.py ./example_data/toy-ppi
'''
if __name__ == "__main__":
    run()
    # run_single_experiment()

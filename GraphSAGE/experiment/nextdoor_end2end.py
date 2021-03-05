import sys,os
from os import path



sys.path.insert(0, os.getcwd())
import time
from graphsage.utils import load_data, run_random_walks
from graphsage.minibatch import NodeMinibatchIterator, EdgeMinibatchIterator, NodeMinibatchIteratorWithKHop
from graphsage.neigh_samplers import UniformNeighborSampler
from graphsage.models import  SAGEInfo
import tensorflow as tf

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


def supervised_epoch_time(PREFIX, batch_size):
    tf.reset_default_graph()
    flags = tf.app.flags
    FLAGS = flags.FLAGS
    del_all_flags(tf.flags.FLAGS)
    from graphsage.supervised_train import train
    FLAGS.train_prefix = PREFIX
    FLAGS.model = 'graphsage_mean'
    FLAGS.batch_size = batch_size
    FLAGS.sigmoid = True
    train_data = load_data(FLAGS.train_prefix)
    time = train(train_data)
    add_to_dict('SEPOCH',time)

def nextdoor_supervised_epoch_time(PREFIX, batch_size):
    tf.reset_default_graph()
    flags = tf.app.flags
    FLAGS = flags.FLAGS
    del_all_flags(tf.flags.FLAGS)
    from graphsage.sampled_supervised_train import  train
    FLAGS.train_prefix = PREFIX
    FLAGS.model = 'graphsage_mean'
    FLAGS.sigmoid = True
    FLAGS.batch_size = batch_size
    train_data = load_data(FLAGS.train_prefix)
    time = train(train_data)
    print("NEXTSEPOCH {}".format(time))
    add_to_dict('NEXTSEPOCH', time)

def del_all_flags(FLAGS):
    flags_dict = FLAGS._flags()
    keys_list = [keys for keys in flags_dict]
    for keys in keys_list:
        FLAGS.__delattr__(keys)


def run():
    global PREFIX
    import sys
    PREFIX = sys.argv[1]
    batchSize = int(sys.argv[2])
    add_to_dict("DATASET",(PREFIX))
    is_available = tf.test.is_gpu_available(
        cuda_only=False, min_cuda_compute_capability=None
    )
    add_to_dict("BATCHSIZE",batchSize)
    add_to_dict ("GPU",is_available)
    G, feats, id_map, walks, class_map = load_data(PREFIX, load_walks=True)
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
    supervised_epoch_time(PREFIX,batchSize)
    nextdoor_supervised_epoch_time(PREFIX,batchSize)
    create_measurement_file()
    print("All Done !!! ")


'''
    how to run !!!
    python experiment/epoch_run_time.py ./example_data/toy-ppi [batch_size]
'''
if __name__ == "__main__":
    # print("hello world")
    run()
    # run_single_experiment()

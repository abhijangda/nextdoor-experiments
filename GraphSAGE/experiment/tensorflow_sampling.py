
import numpy as np
import sys,os
from os import path
sys.path.insert(0, os.getcwd())
def testfun():
    dataset = sys.argv[1]
    fp = open("dataset",'r')
    nodes = {}
    maxv = 0
    for line in fp:
        if line.startswith("#"):
            continue
        a,b = line.split()
        a,b = int(a),int(b)
        if nodes.has_key(a):
            nodes[a].append(b)
        else:
            nodes[a] = [b]
        maxv = max(a,maxv)
        maxv = max(b,maxv)
    max_degree = 128
    adj_matrix =  np.zeros((maxv+1,max_degree))
    train_nodes = []
    for n in nodes:
        train_nodes.append(n)
        neighbors = np.array(nodes[n])
        if len(neighbors) == 0:
            continue
        if len(neighbors) > max_degree:
            neighbors = np.random.choice(neighbors, max_degree, replace=False)
        elif len(neighbors) < max_degree:
            neighbors = np.random.choice(neighbors, max_degree, replace=True)
        adj_matrix[n , :] = neighbors
    from graphsage.neigh_samplers import UniformNeighborSampler
    sampler = UniformNeighborSampler(adj_matrix)
    import tensorflow as tf
    batch =  tf.placeholder(tf.int32, shape=(None,), name='batch')
    sample1 = sampler((batch, 25))
    sample1 = tf.cast(sample1, dtype=tf.int64)
    s_tv = tf.reshape(sample1, [tf.shape(sample1)[0] * 25, ])
    sample2 = sampler((s_tv, 10))
    sess = tf.Session()
    train_nodes =  np.random.permutation(train_nodes)
    import time
    start_time = time.time()
    i = 0
    batchsize = 1
    while  i < len(train_nodes):
        offset = min(len(train_nodes),i+batchsize)
        feed_dict= {batch:train_nodes[i:offset]}
        sess.run(sample2, feed_dict)
        i = i+offset
    end_time = time.time()
    sess.close()
    with open("tf_measurements.txt") as fp:
        fp.write("{} | {}".format(dataset,end_time-start_time))

if __name__ == "__main__":
    run()

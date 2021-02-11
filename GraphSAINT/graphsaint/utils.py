import numpy as np
import json
import pdb
import scipy.sparse
from sklearn.preprocessing import StandardScaler
import os
import yaml
import scipy.sparse as sp
from graphsaint.globals import *
import networkx as nx
import pickle,random

def load_custom_dataset(dataset_str, normalize=True):
    MAX_LABELS = 10
    MAX_FEATURE_SIZE = 8
    if dataset_str == 'patents':
        filename = '/mnt/homes/spolisetty/nextdoor-experiments/datasets/cit-Patents.txt'
        picklefilename = "./cit-Patents.pickle"
    elif dataset_str == 'orkut':
        filename = '/mnt/homes/spolisetty/nextdoor-experiments/datasets/com-orkut.ungraph.txt'
        picklefilename = "./com-orkut.pickle"
    elif dataset_str == 'livejournal':
        filename = '/mnt/homes/spolisetty/nextdoor-experiments/datasets/soc-LiveJournal1.txt'
        picklefilename = "./soc-LiveJournal1.pickle"
    elif dataset_str == "reddit":
        filename = '/mnt/homes/spolisetty/nextdoor-experiments/datasets/reddit_edgelist'
        picklefilename = "./reddit_edgelist.pickle"
    elif dataset_str == "ppi":
        filename = '/mnt/homes/spolisetty/nextdoor-experiments/datasets/ppi_edgelist'
        picklefilename = "./ppi_edgelist.pickle"
    else:
        assert(False)
        
    if (os.path.exists(picklefilename)):
        f = open(picklefilename, 'rb')
        G = pickle.load(f)
        f.close()
        edges = G.edges()
    else:
        edges = []
        G = nx.Graph()
        for line in open(filename):
            if line.startswith('#'):
                continue
            a,b = line.split()
            a,b = int(a),int(b)
            edges += [[a,b]]
        G.add_edges_from(edges)
        print ("Edges Added")
        ################## reorder
        remap = {}
        count = 0
        for i in G.nodes():
            remap[i] = count
            count = count + 1
        G = nx.Graph()
        new_edges = []
        for a,b in edges:
            new_edges += [[remap[a],remap[b]]]
        edges = new_edges
        G.add_edges_from(edges)
        print ("Reorder Done")
        ################## end reorder
        f = open(picklefilename, 'wb')
        pickle.dump(G, f)
        f.close()
    
    N = max(G.nodes())+1
    def coo_format(train_edges):
        r = []
        c = []
        v = []
        for e in train_edges:
            r.append(e[0])
            c.append(e[1])
            v.append(1)
        return r,c,v
    r,c,v = coo_format(edges)
    adj_full = scipy.sparse.coo_matrix((v,(r,c)),shape=(N,N)).astype(np.bool)
    adj_full = adj_full.tocsr()
    # adj_full = scipy.sparse.load_npz('./{}/adj_full.npz'.format(prefix)).astype(np.bool)
    num_data = N 
    degrees = np.zeros(N, dtype=np.int64)
    labels = np.zeros((N,MAX_LABELS), dtype = np.int64)
    idx_train = []
    idx_test = []
    idx_val = []
    class_map = {}
    for s in G:
        r = random.random()
        if r >= .9:
            idx_test += [int(s)]
        elif r >= .8:
            idx_val += [s]
        else:
            idx_train += [s]
        degrees[s] = len(G[s])
        class_map[s] = random.randint(0,MAX_LABELS-1)
        #labels[s][random.randint(0, MAX_LABELS-1)] = 1

    train_data = np.array(idx_train, dtype=np.int32)
    test_data = np.array(idx_test, dtype=np.int32)
    val_data = np.array(idx_val, dtype=np.int32)
    is_train = np.ones((num_data), dtype=np.bool)
    is_train[val_data] = False
    is_train[test_data] = False
    train_edges = [(e[0], e[1]) for e in edges if is_train[e[0]] and is_train[e[1]]]
    r,c,v = coo_format(train_edges)
    adj_train = scipy.sparse.coo_matrix((v,(r,c)),shape=(N,N)).astype(np.bool)
    adj_train = adj_train.tocsr()
    #adj_train = scipy.sparse.load_npz('./{}/adj_train.npz'.format(prefix)).astype(np.bool)
    
    role = {'tr':idx_train,'te':idx_test,'va':idx_val}
    #role = json.load(open('./{}/role.json'.format(prefix)))
    feats = np.random.rand(N,MAX_FEATURE_SIZE)

    #feats = np.load('./{}/feats.npy'.format(prefix))
    
    #class_map = json.load(open('./{}/class_map.json'.format(prefix)))
    #class_map = {int(k):v for k,v in class_map.items()}
    assert len(class_map) == feats.shape[0]
    # ---- normalize feats ---- (Dont touch) 
    #train_nodes = np.array(list(set(adj_train.nonzero()[0])))
    #train_feats = feats[train_nodes]
    #scaler = StandardScaler()
    #scaler.fit(train_feats)
    #feats = scaler.transform(feats)
    # -------------------------
    return adj_full, adj_train, feats, class_map, role

def load_data(prefix, normalize=True):
    """
    Load the various data files residing in the `prefix` directory.
    Files to be loaded:
        adj_full.npz        sparse matrix in CSR format, stored as scipy.sparse.csr_matrix
                            The shape is N by N. Non-zeros in the matrix correspond to all
                            the edges in the full graph. It doesn't matter if the two nodes
                            connected by an edge are training, validation or test nodes.
                            For unweighted graph, the non-zeros are all 1.
        adj_train.npz       sparse matrix in CSR format, stored as a scipy.sparse.csr_matrix
                            The shape is also N by N. However, non-zeros in the matrix only
                            correspond to edges connecting two training nodes. The graph
                            sampler only picks nodes/edges from this adj_train, not adj_full.
                            Therefore, neither the attribute information nor the structural
                            information are revealed during training. Also, note that only
                            a x N rows and cols of adj_train contains non-zeros. For
                            unweighted graph, the non-zeros are all 1.
        role.json           a dict of three keys. Key 'tr' corresponds to the list of all
                              'tr':     list of all training node indices
                              'va':     list of all validation node indices
                              'te':     list of all test node indices
                            Note that in the raw data, nodes may have string-type ID. You
                            need to re-assign numerical ID (0 to N-1) to the nodes, so that
                            you can index into the matrices of adj, features and class labels.
        class_map.json      a dict of length N. Each key is a node index, and each value is
                            either a length C binary list (for multi-class classification)
                            or an integer scalar (0 to C-1, for single-class classification).
        feats.npz           a numpy array of shape N by F. Row i corresponds to the attribute
                            vector of node i.

    Inputs:
        prefix              string, directory containing the above graph related files
        normalize           bool, whether or not to normalize the node features

    Outputs:
        adj_full            scipy sparse CSR (shape N x N, |E| non-zeros), the adj matrix of
                            the full graph, with N being total num of train + val + test nodes.
        adj_train           scipy sparse CSR (shape N x N, |E'| non-zeros), the adj matrix of
                            the training graph. While the shape is the same as adj_full, the
                            rows/cols corresponding to val/test nodes in adj_train are all-zero.
        feats               np array (shape N x f), the node feature matrix, with f being the
                            length of each node feature vector.
        class_map           dict, where key is the node ID and value is the classes this node
                            belongs to.
        role                dict, where keys are: 'tr' for train, 'va' for validation and 'te'
                            for test nodes. The value is the list of IDs of nodes belonging to
                            the train/val/test sets.
    """
    if prefix == 'patents' or prefix == 'orkut' or prefix =='livejournal' or prefix == 'reddit' or prefix =='ppi' :
        print("reached checkpoint")
        return load_custom_dataset(prefix)

    adj_full = scipy.sparse.load_npz('./{}/adj_full.npz'.format(prefix)).astype(np.bool)
    adj_train = scipy.sparse.load_npz('./{}/adj_train.npz'.format(prefix)).astype(np.bool)
    role = json.load(open('./{}/role.json'.format(prefix)))
    feats = np.load('./{}/feats.npy'.format(prefix))
    class_map = json.load(open('./{}/class_map.json'.format(prefix)))
    class_map = {int(k):v for k,v in class_map.items()}
    assert len(class_map) == feats.shape[0]
    # ---- normalize feats ----
    train_nodes = np.array(list(set(adj_train.nonzero()[0])))
    train_feats = feats[train_nodes]
    scaler = StandardScaler()
    scaler.fit(train_feats)
    feats = scaler.transform(feats)
    # -------------------------
    return adj_full, adj_train, feats, class_map, role


def process_graph_data(adj_full, adj_train, feats, class_map, role):
    """
    setup vertex property map for output classes, train/val/test masks, and feats
    """
    num_vertices = adj_full.shape[0]
    if isinstance(list(class_map.values())[0],list):
        num_classes = len(list(class_map.values())[0])
        class_arr = np.zeros((num_vertices, num_classes))
        for k,v in class_map.items():
            class_arr[k] = v
    else:
        num_classes = max(class_map.values()) - min(class_map.values()) + 1
        class_arr = np.zeros((num_vertices, num_classes))
        offset = min(class_map.values())
        for k,v in class_map.items():
            class_arr[k][v-offset] = 1
    return adj_full, adj_train, feats, class_arr, role


def parse_layer_yml(arch_gcn,dim_input):
    """
    Parse the *.yml config file to retrieve the GNN structure.
    """
    num_layers = len(arch_gcn['arch'].split('-'))
    # set default values, then update by arch_gcn
    bias_layer = [arch_gcn['bias']]*num_layers
    act_layer = [arch_gcn['act']]*num_layers
    aggr_layer = [arch_gcn['aggr']]*num_layers
    dims_layer = [arch_gcn['dim']]*num_layers
    order_layer = [int(o) for o in arch_gcn['arch'].split('-')]
    return [dim_input]+dims_layer,order_layer,act_layer,bias_layer,aggr_layer


def parse_n_prepare(flags):
    with open(flags.train_config) as f_train_config:
        train_config = yaml.load(f_train_config)
    arch_gcn = {
        'dim': -1,
        'aggr': 'concat',
        'loss': 'softmax',
        'arch': '1',
        'act': 'I',
        'bias': 'norm'
    }
    arch_gcn.update(train_config['network'][0])
    train_params = {
        'lr': 0.01,
        'weight_decay': 0.,
        'norm_loss': True,
        'norm_aggr': True,
        'q_threshold': 50,
        'q_offset': 0
    }
    train_params.update(train_config['params'][0])
    train_phases = train_config['phase']
    for ph in train_phases:
        assert 'end' in ph
        assert 'sampler' in ph
    print("Loading training data..")
    temp_data = load_data(flags.data_prefix)
    train_data = process_graph_data(*temp_data)
    print("Done loading training data..")
    return train_params,train_phases,train_data,arch_gcn





def log_dir(f_train_config,prefix,git_branch,git_rev,timestamp):
    import getpass
    log_dir = args_global.dir_log+"/log_train/" + prefix.split("/")[-1]
    log_dir += "/{ts}-{model}-{gitrev:s}/".format(
            model='graphsaint',
            gitrev=git_rev.strip(),
            ts=timestamp)
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    if f_train_config != '':
        from shutil import copyfile
        copyfile(f_train_config,'{}/{}'.format(log_dir,f_train_config.split('/')[-1]))
    return log_dir

def sess_dir(dims,train_config,prefix,git_branch,git_rev,timestamp):
    import getpass
    log_dir = "saved_models/" + prefix.split("/")[-1]
    log_dir += "/{ts}-{model}-{gitrev:s}-{layer}/".format(
            model='graphsaint',
            gitrev=git_rev.strip(),
            layer='-'.join(dims),
            ts=timestamp)
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    return sess_dir


def adj_norm(adj, deg=None, sort_indices=True):
    """
    Normalize adj according to the method of rw normalization.
    Note that sym norm is used in the original GCN paper (kipf),
    while rw norm is used in GraphSAGE and some other variants.
    Here we don't perform sym norm since it doesn't seem to
    help with accuracy improvement.

    # Procedure:
    #       1. adj add self-connection --> adj'
    #       2. D' deg matrix from adj'
    #       3. norm by D^{-1} x adj'
    if sort_indices is True, we re-sort the indices of the returned adj
    Note that after 'dot' the indices of a node would be in descending order
    rather than ascending order
    """
    diag_shape = (adj.shape[0],adj.shape[1])
    #adj.setdiag(1)

    D = adj.sum(1).flatten() if deg is None else deg
    
    norm_diag = sp.dia_matrix((1/D,0),shape=diag_shape)
    adj_norm = norm_diag.dot(adj)
    if sort_indices:
        adj_norm.sort_indices()
    return adj_norm




##################
# PRINTING UTILS #
#----------------#

_bcolors = {'header': '\033[95m',
            'blue': '\033[94m',
            'green': '\033[92m',
            'yellow': '\033[93m',
            'red': '\033[91m',
            'bold': '\033[1m',
            'underline': '\033[4m'}


def printf(msg,style=''):
    if not style or style == 'black':
        print(msg)
    else:
        print("{color1}{msg}{color2}".format(color1=_bcolors[style],msg=msg,color2='\033[0m'))

if __name__ == "__main__":
    load_custom_dataset('patents')

#!/usr/bin/env python
# coding: utf-8


from utils import *
from tqdm import tqdm
import argparse
import scipy
import multiprocessing as mp
import time
import warnings, pickle
warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=UserWarning)


parser = argparse.ArgumentParser(description='Training GCN on Cora/CiteSeer/PubMed/Reddit Datasets')

'''
    Dataset arguments
'''
parser.add_argument('--dataset', type=str, default='reddit',
                    help='Dataset name: Cora/CiteSeer/PubMed/Reddit/orkut/patents/livejournal')
parser.add_argument('--nhid', type=int, default=256,
                    help='Hidden state dimension')
parser.add_argument('--epoch_num', type=int, default= 10,
                    help='Number of Epoch')
parser.add_argument('--pool_num', type=int, default= 10,
                    help='Number of Pool')
parser.add_argument('--batch_num', type=int, default= 10,
                    help='Maximum Batch Number')
parser.add_argument('--batch_size', type=int, default=64,
                    help='size of output node in a batch')
parser.add_argument('--n_layers', type=int, default=5,
                    help='Number of GCN layers')
parser.add_argument('--n_iters', type=int, default=1,
                    help='Number of iteration to run on a batch')
parser.add_argument('--n_stops', type=int, default=200,
                    help='Stop after number of batches that f1 dont increase')
parser.add_argument('--samp_num', type=int, default=64,
                    help='Number of sampled nodes per layer')
parser.add_argument('--sample_method', type=str, default='ladies',
                    help='Sampled Algorithms: ladies/fastgcn/full')
parser.add_argument('--cuda', type=int, default=0,
                    help='Avaiable GPU ID')



args = parser.parse_args()



class GraphConvolution(nn.Module):
    def __init__(self, n_in, n_out, bias=True):
        super(GraphConvolution, self).__init__()
        self.n_in  = n_in
        self.n_out = n_out
        self.linear = nn.Linear(n_in,  n_out)
    def forward(self, x, adj):
        out = self.linear(x)
        return F.elu(torch.spmm(adj, out))


class GCN(nn.Module):
    def __init__(self, nfeat, nhid, layers, dropout):
        super(GCN, self).__init__()
        self.layers = layers
        self.nhid = nhid
        self.gcs = nn.ModuleList()
        self.gcs.append(GraphConvolution(nfeat,  nhid))
        self.dropout = nn.Dropout(dropout)
        for i in range(layers-1):
            self.gcs.append(GraphConvolution(nhid,  nhid))
    def forward(self, x, adjs):
        '''
            The difference here with the original GCN implementation is that
            we will receive different adjacency matrix for different layer.
        '''
        for idx in range(len(self.gcs)):
            x = self.dropout(self.gcs[idx](x, adjs[idx]))
        return x

class SuGCN(nn.Module):
    def __init__(self, encoder, num_classes, dropout, inp):
        super(SuGCN, self).__init__()
        self.encoder = encoder
        self.dropout = nn.Dropout(dropout)
        self.linear  = nn.Linear(self.encoder.nhid, num_classes)
    def forward(self, feat, adjs):
        x = self.encoder(feat, adjs)
        x = self.dropout(x)
        x = self.linear(x)
        return x


sampling_time = 0
crit_sampling_time = 0
training_time = 0

def fastgcn_sampler(seed, batch_nodes, samp_num_list, num_nodes, lap_matrix, depth):
    '''
        FastGCN_Sampler: Sample a fixed number of nodes per layer. The sampling probability (importance)
                         is pre-computed based on the global degree (lap_matrix)
    '''
    global sampling_time
    global crit_sampling_time
    t1 = time.time()
    np.random.seed(seed)
    previous_nodes = batch_nodes
    adjs  = []
    #     pre-compute the sampling probability (importance) based on the global degree (lap_matrix)
    pi = np.array(np.sum(lap_matrix.multiply(lap_matrix), axis=0))[0]
    p = pi / np.sum(pi)
    '''
        Sample nodes from top to bottom, based on the pre-computed probability. Then reconstruct the adjacency matrix.
    '''
    for d in range(depth):
        #     row-select the lap_matrix (U) by previously sampled nodes
        U = lap_matrix[previous_nodes , :]
        #t3 = time.time()
        #     sample the next layer's nodes based on the pre-computed probability (p).
        s_num = np.min([np.sum(p > 0), samp_num_list[d]])
        #print(num_nodes)
        t2 = time.time()
        after_nodes = np.random.choice(num_nodes, s_num, p = p, replace = False)
        #t3 = time.time()
        #     col-select the lap_matrix (U), and then devided by the sampled probability for 
        #     unbiased-sampling. Finally, conduct row-normalization to avoid value explosion.         
        adj = (U[: , after_nodes].multiply(1/p[after_nodes]))
        t3 = time.time()

        #t3 = time.time()
        #     Turn the sampled adjacency matrix into a sparse matrix. If implemented by PyG
        #     This sparse matrix can also provide index and value.
        
        adjs += [sparse_mx_to_torch_sparse_tensor(row_normalize(adj))]
        #     Turn the sampled nodes as previous_nodes, recursively conduct sampling.
        #t3 = time.time()
        previous_nodes = after_nodes
        crit_sampling_time += t3-t2
    #     Reverse the sampled probability from bottom to top. Only require input how the lastly sampled nodes.
    t4 = time.time()
    adjs.reverse()
    #t4 = time.time()
    #print(t4-t1,t3-t2)
    sampling_time += t4-t1
    return adjs, previous_nodes, batch_nodes

fastgcn_p = None
fastgcn_p_set = False
sample_number = 0

def nextdoor_fastgcn_sampler(seed, batch_nodes, samp_num_list, num_nodes, lap_matrix, depth):
    enable_nextdoor = True
    global nd, sampling_time, fastgcn_p, fastgcn_p_set, sample_number,crit_sampling_time
    previous_nodes = batch_nodes
    adjs = []
    # p computation can be pushed outside if performance is bad
    samples = []
    
    if (fastgcn_p_set == False):
        pi = np.array(np.sum(lap_matrix.multiply(lap_matrix), axis=0))[0]
        p = pi / np.sum(pi)
        fastgcn_p = p
        fastgcn_p_set = True
    sample_number += 1
    # print("Sampling", len(previous_nodes), samp_num_list[0])
    t1 = time.time()
    for d in range(depth):
        #s_num = np.min([np.sum(p > 0), samp_num_list[d]])
        t3 = time.time()
        after_nodes = nd.sample(0)[sample_number][d] if(enable_nextdoor) else np.random.choice(num_nodes, samp_num_list[d], replace = False)
        t4 = time.time()
        crit_sampling_time += t4-t3
        # print(adj.shape, type(adj), fastgcn_p[after_nodes].shape, type(fastgcn_p[after_nodes]))
        adj = lap_matrix[previous_nodes, : ][:, after_nodes].multiply(1/fastgcn_p[after_nodes])
        
        samples += [(after_nodes, adj)]
        previous_nodes = after_nodes
    previous_nodes = batch_nodes
    t2 = time.time()
    sampling_time += t2-t1
    for d in range(depth):
        after_nodes, adj = samples[d]
        adjs += [sparse_mx_to_torch_sparse_tensor(row_normalize(adj))]
        previous_nodes = after_nodes
    adjs.reverse()
    return adjs, previous_nodes, batch_nodes


def nextdoor_ladies_sampler(seed, batch_nodes, samp_num_list, num_nodes, lap_matrix, depth):
    '''
        LADIES_Sampler: Sample a fixed number of nodes per layer. The sampling probability (importance)
                         is computed adaptively according to the nodes sampled in the upper layer.
    '''
    enable_nextdoor = True
    np.random.seed(seed)
    previous_nodes = batch_nodes
    adjs  = []
    '''
        Sample nodes from top to bottom, based on the probability computed adaptively (layer-dependent).
    '''
    global sampling_time, nd, sample_number

    sample_number += 1
    t1 = time.time()
    for d in range(depth):
        #     row-select the lap_matrix (U) by previously sampled nodes
        U = lap_matrix[previous_nodes , :]
        #     Only use the upper layer's neighborhood to calculate the probability.
        pi = np.array(np.sum(U.multiply(U), axis=0))[0]
        p = pi / np.sum(pi)
        s_num = np.min([np.sum(p > 0), samp_num_list[d]])
        #     sample the next layer's nodes based on the adaptively probability (p).
        after_nodes = nd.sample(0)[sample_number][d] if(enable_nextdoor) else np.random.choice(num_nodes, s_num, p = p, replace = False)
        #     Add output nodes for self-loop
        after_nodes = np.unique(np.concatenate((after_nodes, batch_nodes)))
        #     col-select the lap_matrix (U), and then devided by the sampled probability for 
        #     unbiased-sampling. Finally, conduct row-normalization to avoid value explosion.      
        adj = U[: , after_nodes].multiply(1/p[after_nodes])
        adjs += [sparse_mx_to_torch_sparse_tensor(row_normalize(adj))]
        #     Turn the sampled nodes as previous_nodes, recursively conduct sampling.
        previous_nodes = after_nodes
    #     Reverse the sampled probability from bottom to top. Only require input how the lastly sampled nodes.
    adjs.reverse()
    t2 = time.time()
    sampling_time += t2-t1
    return adjs, previous_nodes, batch_nodes

def ladies_sampler(seed, batch_nodes, samp_num_list, num_nodes, lap_matrix, depth):
    '''
        LADIES_Sampler: Sample a fixed number of nodes per layer. The sampling probability (importance)
                         is computed adaptively according to the nodes sampled in the upper layer.
    '''
    np.random.seed(seed)
    previous_nodes = batch_nodes
    adjs  = []
    '''
        Sample nodes from top to bottom, based on the probability computed adaptively (layer-dependent).
    '''
    global sampling_time,crit_sampling_time
    t1 = time.time()
    for d in range(depth):
        #     row-select the lap_matrix (U) by previously sampled nodes
        U = lap_matrix[previous_nodes , :]
        #     Only use the upper layer's neighborhood to calculate the probability.
        pi = np.array(np.sum(U.multiply(U), axis=0))[0]
        p = pi / np.sum(pi)
        s_num = np.min([np.sum(p > 0), samp_num_list[d]])
        #     sample the next layer's nodes based on the adaptively probability (p).
        _t1 = time.time() #TODO: Right now we consider time spent in only following lines for comparison of application time
        after_nodes = np.random.choice(num_nodes, s_num, p = p, replace = False)
        #     Add output nodes for self-loop
        after_nodes = np.unique(np.concatenate((after_nodes, batch_nodes)))
        _t2 = time.time()
        crit_sampling_time += _t2-_t1
        #     col-select the lap_matrix (U), and then devided by the sampled probability for 
        #     unbiased-sampling. Finally, conduct row-normalization to avoid value explosion.      
        adj = U[: , after_nodes].multiply(1/p[after_nodes])
        adjs += [sparse_mx_to_torch_sparse_tensor(row_normalize(adj))]
        #     Turn the sampled nodes as previous_nodes, recursively conduct sampling.
        previous_nodes = after_nodes
    #     Reverse the sampled probability from bottom to top. Only require input how the lastly sampled nodes.
    adjs.reverse()
    t2 = time.time()
    sampling_time += t2-t1
    return adjs, previous_nodes, batch_nodes

def default_sampler(seed, batch_nodes, samp_num_list, num_nodes, lap_matrix, depth):
    mx = sparse_mx_to_torch_sparse_tensor(lap_matrix)
    return [mx for i in range(depth)], np.arange(num_nodes), batch_nodes
def prepare_data(pool, sampler, process_ids, train_nodes, valid_nodes, samp_num_list, num_nodes, lap_matrix, depth):
    jobs = []
    for _ in process_ids:
        idx = torch.randperm(len(train_nodes))[:args.batch_size]
        batch_nodes = train_nodes[idx]
        p = pool.apply_async(sampler, args=(0, #np.random.randint(2**32 - 1)
        batch_nodes,                                                    samp_num_list, num_nodes, lap_matrix, depth))
        jobs.append(p)
    idx = torch.randperm(len(valid_nodes))[:args.batch_size]
    batch_nodes = valid_nodes[idx]
    p = pool.apply_async(sampler, args=(np.random.randint(2**32 - 1), batch_nodes,                                                samp_num_list, num_nodes, lap_matrix, depth))
    jobs.append(p)
    return jobs
def package_mxl(mxl, device):
    return [torch.sparse.FloatTensor(mx[0], mx[1], mx[2]).to(device) for mx in mxl]



if args.cuda != -1:
    device = torch.device("cuda:" + str(args.cuda))
else:
    device = torch.device("cpu")
    
    
print(args.dataset, args.sample_method)
edges, labels, feat_data, num_classes, train_nodes, valid_nodes, test_nodes = load_data(args.dataset)

lap_matrix_file = args.dataset + "-lap-matrix.pkl"
if os.path.exists(lap_matrix_file):
    f = open(lap_matrix_file, 'rb')
    lap_matrix = pickle.load(f)
    f.close()
else:
    adj_matrix = get_adj(edges, feat_data.shape[0])
    lap_matrix = row_normalize(adj_matrix + sp.eye(adj_matrix.shape[0]))
    f = open(lap_matrix_file, 'wb')
    pickle.dump(lap_matrix, f)
    f.close()
print("lap matrix loaded")

if type(feat_data) == scipy.sparse.lil.lil_matrix:
    feat_data = torch.FloatTensor(feat_data.todense()).to(device) 
else:
    feat_data = torch.FloatTensor(feat_data).to(device)
labels    = torch.LongTensor(labels).to(device) 



if args.sample_method == 'ladies':
    sampler = ladies_sampler
elif args.sample_method == 'fastgcn':
    sampler = fastgcn_sampler
elif args.sample_method == 'full':
    sampler = default_sampler
elif args.sample_method == 'nextdoor_fastgcn':
    sampler = nextdoor_fastgcn_sampler
elif args.sample_method == 'nextdoor_ladies':
    sampler = nextdoor_ladies_sampler

# In[ ]:

training_time = 0
process_ids = np.arange(args.batch_num)
samp_num_list = np.array([args.samp_num, args.samp_num, args.samp_num, args.samp_num, args.samp_num])

from nextdoor_patch import *
nd = NextDoorSamplerFastGCN(args.dataset, args.batch_size, args.dataset, edges, train_nodes, samp_num_list)


asynchronous = False

all_res = []
for oiter in range(5):
    print(feat_data.shape)
    encoder = GCN(nfeat = feat_data.shape[1], nhid=args.nhid, layers=args.n_layers, dropout = 0.2).to(device)
    susage  = SuGCN(encoder = encoder, num_classes=num_classes, dropout=0.5, inp = feat_data.shape[1])
    susage.to(device)

    optimizer = optim.Adam(filter(lambda p : p.requires_grad, susage.parameters()))
    best_val = 0
    best_tst = -1
    cnt = 0
    times = []
    res   = []
    print('-' * 10)
    print(len(feat_data))
    end_to_end_t1 = time.time()
    if (asynchronous):
        pool = mp.Pool(args.pool_num)
        jobs = prepare_data(pool, sampler, process_ids, train_nodes, valid_nodes, samp_num_list, len(feat_data), lap_matrix, args.n_layers)

    for epoch in np.arange(args.epoch_num):
        susage.train()
        train_losses = []
        batches = len(train_nodes)//args.batch_size
        print("num batches", batches)
        if (not asynchronous):
            ######################### Synchronous Version ########################
            useMP = False
            if False:
                with mp.Pool(processes=args.pool_num) as pool:
                    idx = torch.randperm(len(train_nodes))[:args.batch_size]
                    batch_nodes = train_nodes[idx]
                    sampleArgs = (np.random.randint(2**32 - 1), batch_nodes, \
                        samp_num_list, len(feat_data), lap_matrix, args.n_layers)
                    jobs = [pool.apply_async(sampler, sampleArgs) for b in range(args.batch_num)]
                    train_data = [job.get() for job in jobs[:-1]]
                    valid_data = jobs[-1].get()

            else:
                idx = torch.randperm(len(train_nodes))[:args.batch_size]
                batch_nodes = train_nodes[idx]
                train_data = [sampler(np.random.randint(2**32 - 1), batch_nodes, \
                        samp_num_list, len(feat_data), lap_matrix, args.n_layers) for b in range(args.batch_num)]
                idx = torch.randperm(len(valid_nodes))[:args.batch_size]
                batch_nodes = valid_nodes[idx]
                valid_data = sampler(np.random.randint(2**32 - 1), batch_nodes, \
                        samp_num_list, len(feat_data), lap_matrix, args.n_layers)
        else:
            ########### Asynchronous version ###############
            train_data = [job.get() for job in jobs[:-1]]
            valid_data = jobs[-1].get()
            pool.close()
            pool.join()
            pool = mp.Pool(args.pool_num)
            '''
                #Use CPU-GPU cooperation to reduce the overhead for sampling. (conduct sampling while training)
            '''
            jobs = prepare_data(pool, sampler, process_ids, train_nodes, valid_nodes, samp_num_list, len(feat_data), lap_matrix, args.n_layers)
        print("len(train_data)", len(train_data))
        t0 = time.time()
        for _iter in range(args.n_iters):
            for adjs, input_nodes, output_nodes in train_data:    
                adjs = package_mxl(adjs, device)
                optimizer.zero_grad()
                t1 = time.time()
                susage.train()
                output = susage.forward(feat_data[input_nodes], adjs)
                if args.sample_method == 'full':
                    output = output[output_nodes]
                loss_train = F.cross_entropy(output, labels[output_nodes])
                loss_train.backward()
                torch.nn.utils.clip_grad_norm_(susage.parameters(), 0.2)
                optimizer.step()
                times += [time.time() - t1]
                train_losses += [loss_train.detach().tolist()]
                del loss_train
        susage.eval()
        adjs, input_nodes, output_nodes = valid_data
        adjs = package_mxl(adjs, device)
        output = susage.forward(feat_data[input_nodes], adjs)
        if args.sample_method == 'full':
            output = output[output_nodes]
        loss_valid = F.cross_entropy(output, labels[output_nodes]).detach().tolist()
        valid_f1 = f1_score(output.argmax(dim=1).cpu(), labels[output_nodes].cpu(), average='micro')
        print(("Epoch: %d (%.1fs) Train Loss: %.2f    Valid Loss: %.2f Valid F1: %.3f") %                   (epoch, np.sum(times), np.average(train_losses), loss_valid, valid_f1))
        if valid_f1 > best_val + 1e-2:
            best_val = valid_f1
            torch.save(susage, './save/best_model.pt')
            cnt = 0
        else:
            cnt += 1
        # if cnt == args.n_stops // args.batch_num:
        #     break
        t2 = time.time()
        training_time += t2-t0
    
    end_to_end_t2 = time.time()
    print("end_to_end_time", end_to_end_t2 - end_to_end_t1)
    print("training_time",training_time)
    print("sampling_time",sampling_time)
    print("crit_sampling_time", crit_sampling_time)
    print("Per Iteration Application Sampling time for %d nodes:"%(len(train_nodes)) ,crit_sampling_time/args.batch_num * batches / (100 if (len(train_nodes) > 1000000) else 1)/args.epoch_num)
    break
    best_model = torch.load('./save/best_model.pt')
    best_model.eval()
    test_f1s = []
    '''
    If using batch sampling for inference:
    '''
    #     for b in np.arange(len(test_nodes) // args.batch_size):
    #         batch_nodes = test_nodes[b * args.batch_size : (b+1) * args.batch_size]
    #         adjs, input_nodes, output_nodes = sampler(np.random.randint(2**32 - 1), batch_nodes,
    #                                     samp_num_list * 20, len(feat_data), lap_matrix, args.n_layers)
    #         adjs = package_mxl(adjs, device)
    #         output = best_model.forward(feat_data[input_nodes], adjs)[output_nodes]
    #         test_f1 = f1_score(output.argmax(dim=1).cpu(), labels[output_nodes].cpu(), average='micro')
    #         test_f1s += [test_f1]
    
    '''
    If using full-batch inference:
    '''
    batch_nodes = test_nodes
    adjs, input_nodes, output_nodes = default_sampler(np.random.randint(2**32 - 1), batch_nodes,
                                    samp_num_list * 20, len(feat_data), lap_matrix, args.n_layers)
    adjs = package_mxl(adjs, device)
    output = best_model.forward(feat_data[input_nodes], adjs)[output_nodes]
    test_f1s = [f1_score(output.argmax(dim=1).cpu(), labels[output_nodes].cpu(), average='micro')]
    
    print('Iteration: %d, Test F1: %.3f' % (oiter, np.average(test_f1s)))
    print("training_time",training_time)
    print("sampling_time",sampling_time)
    print("crit_time",crit_sampling_time)
'''
    Visualize the train-test curve
'''

# dt = pd.DataFrame(all_res, columns=['f1-score', 'batch', 'type'])
# sb.lineplot(data = dt, x='batch', y='f1-score', hue='type')
# plt.legend(loc='lower right')
# plt.show()


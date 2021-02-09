from utils import *
import metis
from partition_utils import partition_graph


def CalculateThreshold(candidatesArray, sampleSize, sumSmall=0, nLarge=0):
    candidate = candidatesArray[candidatesArray > 0][0]
    smallArray = candidatesArray[candidatesArray < candidate]
    largeArray = candidatesArray[candidatesArray > candidate]
    equalArray = candidatesArray[candidatesArray == candidate]
    curSampleSize = (sum(smallArray) + sumSmall) / candidate + \
        len(largeArray) + nLarge + len(equalArray)
    if curSampleSize < sampleSize:
        if len(smallArray) == 0:
            return sumSmall/(sampleSize-nLarge-len(largeArray)-1)
        else:
            nLarge = nLarge + len(largeArray)+len(equalArray)
            return CalculateThreshold(smallArray, sampleSize, sumSmall, nLarge)
    else:
        if len(largeArray) == 0:
            return (sumSmall + sum(smallArray) + sum(equalArray))/(sampleSize-nLarge)
        else:
            sumSmall = sumSmall + sum(smallArray) + sum(equalArray)
            return CalculateThreshold(largeArray, sampleSize, sumSmall, nLarge)
        
class base_sampler:
    def __init__(self, adj_matrix, train_nodes):
        assert(adj_matrix.diagonal().sum() == 0)  # make sure diagnal is zero
        # make sure is symmetric
        assert((adj_matrix != adj_matrix.T).nnz == 0)
        self.adj_matrix = adj_matrix
        self.train_nodes = train_nodes
        self.lap_matrix = normalize(adj_matrix + sp.eye(adj_matrix.shape[0]))
        self.lap_matrix_sq = self.lap_matrix.multiply(self.lap_matrix)
        self.lap_norm = np.array(np.sum(self.lap_matrix, axis=1))
        
    def full_batch(self, batch_nodes, num_nodes, depth):
        adjs = [sparse_mx_to_torch_sparse_tensor(
            self.lap_matrix) for _ in range(depth)]
        input_nodes = np.arange(num_nodes)
        sampled_nodes = [np.arange(num_nodes) for _ in range(depth)]
        return adjs, input_nodes, sampled_nodes

    def large_batch(self, batch_nodes, num_nodes, depth):
        previous_nodes = batch_nodes
        sampled_nodes = []
        adjs = []
        for d in range(depth):
            U = self.lap_matrix[previous_nodes, :]
            after_nodes = []
            for U_row in U:
                indices = U_row.indices
                after_nodes.append(indices)
            after_nodes = np.unique(np.concatenate(after_nodes))
            after_nodes = np.concatenate(
                [previous_nodes, np.setdiff1d(after_nodes, previous_nodes)])
            adj = U[:, after_nodes]
            adjs += [sparse_mx_to_torch_sparse_tensor(adj)]
            sampled_nodes.append(previous_nodes)
            previous_nodes = after_nodes
        adjs.reverse()
        sampled_nodes.reverse()
        return adjs, previous_nodes, sampled_nodes
    
    def row_norm(self, adj, select):
        adj = row_normalize(adj)
        adj = adj.multiply(self.lap_norm[select]) 
        # adj_norm = np.array(np.sum(adj, axis=1))
        # adj = adj.multiply(self.lap_norm[select]/adj_norm) 
        return adj
        
class fastgcn_sampler(base_sampler):
    def __init__(self, adj_matrix, train_nodes):
        assert(adj_matrix.diagonal().sum() == 0)  # make sure diagnal is zero
        # make sure is symmetric
        assert((adj_matrix != adj_matrix.T).nnz == 0)
        self.adj_matrix = adj_matrix
        self.train_nodes = train_nodes
        self.lap_matrix = normalize(adj_matrix + sp.eye(adj_matrix.shape[0]))
        self.lap_matrix_sq = self.lap_matrix.multiply(self.lap_matrix)

    def mini_batch(self, seed, batch_nodes, probs_nodes, samp_num_list, num_nodes, adj_matrix, depth):
        np.random.seed(seed)
        previous_nodes = batch_nodes
        sampled_nodes = []
        adjs = []
        pi = np.array(np.sum(self.lap_matrix_sq, axis=0))[0]
        p = pi / np.sum(pi)
        for d in range(depth):
            U = self.lap_matrix[previous_nodes, :]
            s_num = np.min([np.sum(p > 0), samp_num_list[d]])
            after_nodes = np.random.choice(
                num_nodes, s_num, p=p, replace=False)
            after_nodes = np.unique(after_nodes)
            adj = U[:, after_nodes].multiply(1/p[after_nodes]/num_nodes)
            adjs += [sparse_mx_to_torch_sparse_tensor(row_normalize(adj))]
            sampled_nodes += [previous_nodes]
            previous_nodes = after_nodes
        sampled_nodes.reverse()
        adjs.reverse()
        return adjs, previous_nodes, batch_nodes, probs_nodes, sampled_nodes

class ladies_sampler(base_sampler):
    def __init__(self, adj_matrix, train_nodes):
        assert(adj_matrix.diagonal().sum() == 0)  # make sure diagnal is zero
        # make sure is symmetric
        assert((adj_matrix != adj_matrix.T).nnz == 0)
        self.adj_matrix = adj_matrix
        self.train_nodes = train_nodes
        self.lap_matrix = normalize(adj_matrix + sp.eye(adj_matrix.shape[0]))
        self.lap_matrix_sq = self.lap_matrix.multiply(self.lap_matrix)

    def mini_batch(self, seed, batch_nodes, probs_nodes, samp_num_list, num_nodes, adj_matrix, depth):
        np.random.seed(seed)
        previous_nodes = batch_nodes
        sampled_nodes = []
        adjs = []
        for d in range(depth):
            U = self.lap_matrix[previous_nodes, :]
            pi = np.array(
                np.sum(self.lap_matrix_sq[previous_nodes, :], axis=0))[0]
            p = pi / np.sum(pi)
            s_num = np.min([np.sum(p > 0), samp_num_list[d]])
            after_nodes = np.random.choice(
                num_nodes, s_num, p=p, replace=False)
            after_nodes = np.unique(np.concatenate((after_nodes, batch_nodes)))
            adj = U[:, after_nodes].multiply(1/p[after_nodes])
            adj = row_normalize(adj)
            adjs += [sparse_mx_to_torch_sparse_tensor(adj)]
            sampled_nodes += [previous_nodes]
            previous_nodes = after_nodes
        sampled_nodes.reverse()
        adjs.reverse()
        return adjs, previous_nodes, batch_nodes, probs_nodes, sampled_nodes
    
    def mini_batch_ld(self, seed, batch_nodes, probs_nodes, samp_num_list, num_nodes, adj_matrix, depth):
        np.random.seed(seed)
        previous_nodes = batch_nodes
        sampled_nodes = []
        adjs = []
        for d in range(depth):
            U = self.lap_matrix[previous_nodes, :]
            pi = np.array(
                np.sum(self.lap_matrix_sq[previous_nodes, :], axis=0))[0]
            p = pi / np.sum(pi)
            s_num = np.min([np.sum(p > 0), samp_num_list[d]])
            after_nodes = np.random.choice(
                num_nodes, s_num, p=p, replace=False)
            after_nodes = np.unique(after_nodes)
            
            after_nodes = np.concatenate(
                [previous_nodes, np.setdiff1d(after_nodes, previous_nodes)])
            
            adj = U[:, after_nodes].multiply(1/p[after_nodes])
            adj = row_normalize(adj)
            
            adjs += [sparse_mx_to_torch_sparse_tensor(adj)]
            sampled_nodes += [previous_nodes]
            previous_nodes = after_nodes
        sampled_nodes.reverse()
        adjs.reverse()
        return adjs, previous_nodes, batch_nodes, probs_nodes, sampled_nodes

class cluster_sampler(base_sampler):
    def __init__(self, adj_matrix, train_nodes, num_clusters):
        assert(adj_matrix.diagonal().sum() == 0)  # make sure diagnal is zero
        # make sure is symmetric
        assert((adj_matrix != adj_matrix.T).nnz == 0)
        self.adj_matrix = adj_matrix
        self.lap_matrix = normalize(adj_matrix+sp.eye(adj_matrix.shape[0]))
        self.train_nodes = train_nodes
        self.num_clusters = num_clusters
        self.parts = partition_graph(
            adj_matrix, train_nodes, num_clusters)

    def sample_subgraph(self, seed, size=1):
        np.random.seed(seed)
        select = np.random.choice(self.num_clusters, size, replace=False)
        select = [self.parts[i] for i in select]
        batch_nodes = np.concatenate(select)
        return batch_nodes

    def mini_batch(self, seed, batch_nodes, probs_nodes, samp_num_list, num_nodes, adj_matrix, depth):
        bsize = samp_num_list[0]
        batch_nodes = self.sample_subgraph(seed, bsize)

        sampled_nodes = []

        adj = self.adj_matrix[batch_nodes, :][:, batch_nodes]
        adj = normalize(adj+sp.eye(adj.shape[0]))
        adjs = []
        for d in range(depth):
            adjs += [sparse_mx_to_torch_sparse_tensor(adj)]
            sampled_nodes.append(batch_nodes)
        adjs.reverse()
        sampled_nodes.reverse()
        return adjs, batch_nodes, batch_nodes, probs_nodes, sampled_nodes

class graphsage_sampler(base_sampler):
    def __init__(self, adj_matrix, train_nodes):
        assert(adj_matrix.diagonal().sum() == 0)  # make sure diagnal is zero
        # make sure is symmetric
        assert((adj_matrix != adj_matrix.T).nnz == 0)
        self.adj_matrix = adj_matrix
        self.train_nodes = train_nodes
        self.lap_matrix = normalize(
            adj_matrix + sp.eye(adj_matrix.shape[0]))
        self.lap_norm = np.array(np.sum(self.lap_matrix, axis=1))

    def mini_batch(self, seed, batch_nodes, probs_nodes, samp_num_list, num_nodes, adj_matrix, depth):
        np.random.seed(seed)
        sampled_nodes = []
        previous_nodes = batch_nodes
        adjs = []
        for d in range(depth):
            U = self.lap_matrix[previous_nodes, :]
            after_nodes = []
            for U_row in U:
                indices = U_row.indices
                sampled_indices = np.random.choice(
                    indices, samp_num_list[d], replace=True)
                after_nodes.append(sampled_indices)
            after_nodes = np.unique(np.concatenate(after_nodes))
            after_nodes = np.concatenate(
                [previous_nodes, np.setdiff1d(after_nodes, previous_nodes)])
            adj = U[:, after_nodes]
            adj = self.row_norm(adj, previous_nodes)
            adjs += [sparse_mx_to_torch_sparse_tensor(adj)]
            sampled_nodes.append(previous_nodes)
            previous_nodes = after_nodes
        adjs.reverse()
        sampled_nodes.reverse()
        return adjs, previous_nodes, batch_nodes, probs_nodes, sampled_nodes

class vrgcn_sampler(base_sampler):
    def __init__(self, adj_matrix, train_nodes):
        assert(adj_matrix.diagonal().sum() == 0)  # make sure diagnal is zero
        # make sure is symmetric
        assert((adj_matrix != adj_matrix.T).nnz == 0)
        self.adj_matrix = adj_matrix
        self.lap_matrix = normalize(
            adj_matrix + sp.eye(adj_matrix.shape[0]))
        self.train_nodes = train_nodes
        self.lap_norm = np.array(np.sum(self.lap_matrix, axis=1))

    def mini_batch(self, seed, batch_nodes, probs_nodes, samp_num_list, num_nodes, adj_matrix, depth):
        np.random.seed(seed)
        sampled_nodes = []
        exact_input_nodes = []
        previous_nodes = batch_nodes
        adjs = []
        adjs_exact = []

        for d in range(depth):
            U = self.lap_matrix[previous_nodes, :]
            after_nodes = []
            after_nodes_exact = []
            for U_row in U:
                indices = U_row.indices
                s_num = min(len(indices), samp_num_list[d])
                sampled_indices = np.random.choice(
                    indices, s_num, replace=False)
                after_nodes.append(sampled_indices)
                after_nodes_exact.append(indices)
            after_nodes = np.unique(np.concatenate(after_nodes))
            after_nodes = np.concatenate(
                [previous_nodes, np.setdiff1d(after_nodes, previous_nodes)])
            after_nodes_exact = np.unique(np.concatenate(after_nodes_exact))
            after_nodes_exact = np.concatenate(
                [previous_nodes, np.setdiff1d(after_nodes_exact, previous_nodes)])
            adj = U[:, after_nodes]
            adj = self.row_norm(adj, previous_nodes) 
            adjs += [sparse_mx_to_torch_sparse_tensor(adj)]
            
            adj_exact = U[:, after_nodes_exact]
            adjs_exact += [sparse_mx_to_torch_sparse_tensor(adj_exact)]
            sampled_nodes.append(previous_nodes)
            exact_input_nodes.append(after_nodes_exact)
            previous_nodes = after_nodes
        adjs.reverse()
        sampled_nodes.reverse()
        adjs_exact.reverse()
        exact_input_nodes.reverse()
        return adjs, adjs_exact, previous_nodes, batch_nodes, probs_nodes, sampled_nodes, exact_input_nodes

class graphsaint_sampler(base_sampler):
    def __init__(self, adj_matrix, train_nodes, node_budget):
        assert(adj_matrix.diagonal().sum() == 0)  # make sure diagnal is zero
        # make sure is symmetric
        assert((adj_matrix != adj_matrix.T).nnz == 0)
        self.adj_matrix = adj_matrix
        self.lap_matrix = normalize(adj_matrix + sp.eye(adj_matrix.shape[0]))

        adj_matrix_train = adj_matrix[train_nodes, :][:, train_nodes]
        lap_matrix_train = normalize(
            adj_matrix_train + sp.eye(adj_matrix_train.shape[0]))
        self.lap_matrix_train = lap_matrix_train
        lap_matrix_train_sq = lap_matrix_train.multiply(lap_matrix_train)
        p = np.array(np.sum(lap_matrix_train_sq, axis=0))[0]
        self.sample_prob = node_budget*p/p.sum()
        self.train_nodes = train_nodes
        self.node_budget = node_budget
        
        self.lap_norm = np.array(np.sum(self.lap_matrix, axis=1))

    def mini_batch(self, seed, batch_nodes, probs_nodes, samp_num_list, num_nodes, adj_matrix, depth):
        np.random.seed(seed)
        sample_mask = np.random.uniform(
            0, 1, len(self.train_nodes)) <= self.sample_prob
        probs_nodes = self.sample_prob[sample_mask]
        batch_nodes = self.train_nodes[sample_mask]
        adj = self.lap_matrix[batch_nodes, :][:, batch_nodes].multiply(1/probs_nodes)
        adj = self.row_norm(adj, batch_nodes)
        
        adjs = []
        sampled_nodes = []
        for d in range(depth):
            adjs += [sparse_mx_to_torch_sparse_tensor(adj)]
            sampled_nodes.append(batch_nodes)
        adjs.reverse()
        sampled_nodes.reverse()

        return adjs, batch_nodes, batch_nodes, probs_nodes*len(self.train_nodes), sampled_nodes

class subgraph_sampler(base_sampler):
    def __init__(self, adj_matrix, train_nodes):
        assert(adj_matrix.diagonal().sum() == 0)  # make sure diagnal is zero
        # make sure is symmetric
        assert((adj_matrix != adj_matrix.T).nnz == 0)
        self.adj_matrix = adj_matrix.tocoo()
        self.lap_matrix = normalize(adj_matrix + sp.eye(adj_matrix.shape[0]))
        self.train_nodes = train_nodes

    def dropedge(self, percent=0.8):
        nnz = self.adj_matrix.nnz
        perm = np.random.permutation(nnz)
        preserve_nnz = int(nnz*percent)
        perm = perm[:preserve_nnz]
        adj_matrix = sp.csr_matrix((self.adj_matrix.data[perm],
                               (self.adj_matrix.row[perm],
                                self.adj_matrix.col[perm])),
                                shape=self.adj_matrix.shape)
        lap_matrix = normalize(adj_matrix + sp.eye(adj_matrix.shape[0]))
        return lap_matrix

    def mini_batch(self, seed, batch_nodes, probs_nodes, samp_num_list, num_nodes, adj_matrix, depth):
        # lap_matrix = self.dropedge(percent=0.8)
        # adj = lap_matrix[batch_nodes, :][:, batch_nodes]
        # U = lap_matrix[batch_nodes, :]
        adj = self.lap_matrix[batch_nodes, :][:, batch_nodes]
        U = self.lap_matrix[batch_nodes, :]
        
        is_neighbor = np.array(np.sum(U, axis=0))[0]>0
        neighbors = np.arange(len(is_neighbor))[is_neighbor]
        neighbors = np.setdiff1d(neighbors, batch_nodes)
        after_nodes_exact = np.concatenate([batch_nodes, neighbors])
        adj_exact = U[:, after_nodes_exact]
        
#         for U_row in U:
#             indices = U_row.indices
#             after_nodes.append(indices)
#         after_nodes = np.unique(np.concatenate(after_nodes))
#         after_nodes = np.setdiff1d(after_nodes, batch_nodes)
#         after_nodes_exact = np.concatenate([batch_nodes, after_nodes])
#         adj_exact = U[:, after_nodes_exact]

        adjs = []
        adjs_exact = []
        sampled_nodes = []
        input_nodes_exact = []
        for d in range(depth):
            adjs += [sparse_mx_to_torch_sparse_tensor(adj)]
            adjs_exact += [sparse_mx_to_torch_sparse_tensor(adj_exact)]
            sampled_nodes.append(batch_nodes)
            input_nodes_exact.append(neighbors)
        adjs.reverse()
        adjs_exact.reverse()
        sampled_nodes.reverse()
        input_nodes_exact.reverse()

        return adjs, adjs_exact, batch_nodes, batch_nodes, probs_nodes, sampled_nodes, input_nodes_exact

class exact_sampler(base_sampler):
    def __init__(self, adj_matrix, train_nodes):
        assert(adj_matrix.diagonal().sum() == 0)  # make sure diagnal is zero
        # make sure is symmetric
        assert((adj_matrix != adj_matrix.T).nnz == 0)
        self.adj_matrix = adj_matrix
        self.train_nodes = train_nodes
        self.lap_matrix = normalize(adj_matrix + sp.eye(adj_matrix.shape[0]))

    def mini_batch(self, seed, batch_nodes, probs_nodes, samp_num_list, num_nodes, adj_matrix, depth):
        sampled_nodes = []
        previous_nodes = batch_nodes
        adjs = []
        for d in range(depth):
            U = self.lap_matrix[previous_nodes, :]
            after_nodes = []
            for U_row in U:
                indices = U_row.indices
                after_nodes.append(indices)
            after_nodes = np.unique(np.concatenate(after_nodes))
            after_nodes = np.concatenate(
                [previous_nodes, np.setdiff1d(after_nodes, previous_nodes)])
            adj = U[:, after_nodes]
            adjs += [sparse_mx_to_torch_sparse_tensor(adj)]
            sampled_nodes.append(previous_nodes)
            previous_nodes = after_nodes
        adjs.reverse()
        sampled_nodes.reverse()
        return adjs, previous_nodes, batch_nodes, probs_nodes, sampled_nodes
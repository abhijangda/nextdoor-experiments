import multiprocessing as mp
from samplers import CalculateThreshold
from optimizers import package_mxl
from samplers import ladies_sampler, vrgcn_sampler, graphsage_sampler, fastgcn_sampler, exact_sampler, subgraph_sampler, cluster_sampler, graphsaint_sampler
import numpy as np
    
def prepare_data(pool, sampler, process_ids, train_nodes, train_nodes_p, samp_num_list, num_nodes, adj_matrix, depth, is_ratio=1.0):
    num_train_nodes = len(train_nodes)
    jobs = []
    for p_id in process_ids:
        sample_mask = np.random.uniform(0, 1, num_train_nodes)<= train_nodes_p
        probs_nodes = train_nodes_p[sample_mask] * len(train_nodes) * is_ratio
        batch_nodes = train_nodes[sample_mask]
        p = pool.apply_async(sampler, args=(np.random.randint(2**32 - 1), batch_nodes, probs_nodes,
                                            samp_num_list, num_nodes, adj_matrix, depth))
        jobs.append(p)
    return jobs


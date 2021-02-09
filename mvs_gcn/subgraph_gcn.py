from utils import *
from packages import *
from optimizers import sgd_step, variance_reduced_step, boost_step
from forward_wrapper import ForwardWrapper
from model import GCN, GraphSageGCN

"""
ClusterGCN
"""


def clustergcn(feat_data, labels, adj_matrix, train_nodes, valid_nodes, test_nodes,  args, device, concat=False):
    if args.dataset=='ppi':
        samp_num_list = np.array([1 for _ in range(args.n_layers)])
        cluster_num = 50
    else:
        samp_num_list = np.array([int(args.batch_size/128) for _ in range(args.n_layers)])
        cluster_num = int(len(train_nodes)/128)
            
    # use multiprocess sample data
    process_ids = np.arange(args.batch_num)
    
    print(cluster_num)
    cluster_sampler_ = cluster_sampler(adj_matrix, train_nodes, cluster_num)

    if concat:
        susage = GraphSageGCN(nfeat=feat_data.shape[1], nhid=args.nhid, num_classes=args.num_classes,
                     layers=args.n_layers, dropout=args.dropout, multi_class=args.multi_class).to(device)
    else:
        susage = GCN(nfeat=feat_data.shape[1], nhid=args.nhid, num_classes=args.num_classes,
                              layers=args.n_layers, dropout=args.dropout, multi_class=args.multi_class).to(device)
    susage.to(device)
    print(susage)

    optimizer = optim.Adam(susage.parameters(), lr=0.01)

    adjs_full, input_nodes_full, sampled_nodes_full = cluster_sampler_.full_batch(
        train_nodes, len(feat_data), args.n_layers)
    adjs_full = package_mxl(adjs_full, device)

    best_model = copy.deepcopy(susage)
    susage.zero_grad()
    cur_test_loss = susage.calculate_loss_grad(
        feat_data, adjs_full, labels, valid_nodes)

    best_val, cnt = 0, 0

    loss_train = [cur_test_loss]
    loss_test = [cur_test_loss]
    grad_variance_all = []
    loss_train_all = [cur_test_loss]
    times = []
    data_prepare_times = []

    for epoch in np.arange(args.epoch_num):

        train_nodes_p = args.batch_size * \
            np.ones_like(train_nodes)/len(train_nodes)

        # prepare train data
        tp0 = time.time()
        pool = mp.Pool(args.pool_num)
        jobs = prepare_data(pool, cluster_sampler_.mini_batch, process_ids, train_nodes, train_nodes_p, samp_num_list, len(feat_data),
                            adj_matrix, args.n_layers)
        # fetch train data
        train_data = [job.get() for job in jobs]
        pool.close()
        pool.join()
        tp1 = time.time()
        data_prepare_times += [tp1-tp0]
        
        inner_loop_num = args.batch_num

        t0 = time.time()
        cur_train_loss, cur_train_loss_all, grad_variance = sgd_step(susage, optimizer, feat_data, labels,
                                                                     train_nodes, valid_nodes,
                                                                     adjs_full, train_data, inner_loop_num, device,
                                                                     calculate_grad_vars=bool(args.show_grad_norm) if epoch<int(200/args.batch_num) else False)
        t1 = time.time()

        times += [t1-t0]
        print('sgcn run time per epoch is %0.3f' % (t1-t0))
        loss_train_all.extend(cur_train_loss_all)
        grad_variance_all.extend(grad_variance)
        # calculate test loss
        susage.eval()

        susage.zero_grad()
        cur_test_loss = susage.calculate_loss_grad(
            feat_data, adjs_full, labels, valid_nodes)
        val_f1 = susage.calculate_f1(feat_data, adjs_full, labels, valid_nodes)

        if val_f1 > best_val:
            best_model = copy.deepcopy(susage)
        if val_f1 > best_val + 1e-2:
            best_val = val_f1
            cnt = 0
        else:
            cnt += 1
        if cnt == args.n_stops // args.batch_num:
            break

        loss_train.append(cur_train_loss)
        loss_test.append(cur_test_loss)

        # print progress
        print('Epoch: ', epoch,
              '| train loss: %.8f' % cur_train_loss,
              '| test loss: %.8f' % cur_test_loss,
              '| test f1: %.8f' % val_f1)

    f1_score_test = best_model.calculate_f1(
        feat_data, adjs_full, labels, test_nodes)
    
    if bool(args.show_grad_norm):
        times, data_prepare_times = times[int(200/args.batch_num):], data_prepare_times[int(200/args.batch_num):]
    print('Average training time is %0.3f' % np.mean(times))
    print('Average data prepare time is %0.3f'%np.mean(data_prepare_times))
    return best_model, loss_train, loss_test, loss_train_all, f1_score_test, grad_variance_all


"""
GraphSaint
"""

def graphsaint(feat_data, labels, adj_matrix, train_nodes, valid_nodes, test_nodes,  args, device, concat=False):
    samp_num_list = np.array([args.samp_num for _ in range(args.n_layers)])

    # use multiprocess sample data
    process_ids = np.arange(args.batch_num)

    graphsaint_sampler_ = graphsaint_sampler(adj_matrix, train_nodes, node_budget=args.batch_size)

    if concat:
        susage = GraphSageGCN(nfeat=feat_data.shape[1], nhid=args.nhid, num_classes=args.num_classes,
                          layers=args.n_layers, dropout=args.dropout, multi_class=args.multi_class).to(device)
    else:
        susage = GCN(nfeat=feat_data.shape[1], nhid=args.nhid, num_classes=args.num_classes,
                          layers=args.n_layers, dropout=args.dropout, multi_class=args.multi_class).to(device)

    susage.to(device)
    print(susage)

    optimizer = optim.Adam(susage.parameters())

    adjs_full, input_nodes_full, sampled_nodes_full = graphsaint_sampler_.full_batch(
        train_nodes, len(feat_data), args.n_layers)
    adjs_full = package_mxl(adjs_full, device)

    best_model = copy.deepcopy(susage)
    susage.zero_grad()
    cur_test_loss = susage.calculate_loss_grad(
        feat_data, adjs_full, labels, valid_nodes)

    best_val, cnt = 0, 0

    loss_train = [cur_test_loss]
    loss_test = [cur_test_loss]
    grad_variance_all = []
    loss_train_all = [cur_test_loss]
    times = []
    data_prepare_times = []

    for epoch in np.arange(args.epoch_num):

        train_nodes_p = args.batch_size * \
            np.ones_like(train_nodes)/len(train_nodes)

        # prepare train data
        tp0 = time.time()
        pool = mp.Pool(args.pool_num)
        jobs = prepare_data(pool, graphsaint_sampler_.mini_batch, process_ids, train_nodes, train_nodes_p, samp_num_list, len(feat_data),
                            adj_matrix, args.n_layers)
        # fetch train data
        train_data = [job.get() for job in jobs]
        pool.close()
        pool.join()
        tp1 = time.time()
        data_prepare_times += [tp1-tp0]
        
        inner_loop_num = args.batch_num

        t2 = time.time()
        
        cur_train_loss, cur_train_loss_all, grad_variance = boost_step(susage, optimizer, feat_data, labels,
                                                                     train_nodes, valid_nodes,
                                                                     adjs_full, train_data, inner_loop_num, device,
                                                                     calculate_grad_vars=bool(args.show_grad_norm) if epoch<int(200/args.batch_num) else False)
        t3 = time.time()

        times += [t3-t2]
        print('mvs_gcn_plus run time per epoch is %0.3f' % (t3-t2))

        loss_train_all.extend(cur_train_loss_all)
        grad_variance_all.extend(grad_variance)
        # calculate test loss
        susage.eval()

        susage.zero_grad()
        cur_test_loss = susage.calculate_loss_grad(
            feat_data, adjs_full, labels, valid_nodes)
        val_f1 = susage.calculate_f1(feat_data, adjs_full, labels, valid_nodes)

        if val_f1 > best_val:
            best_model = copy.deepcopy(susage)
        if val_f1 > best_val + 1e-2:
            best_val = val_f1
            cnt = 0
        else:
            cnt += 1
        if cnt == args.n_stops // args.batch_num:
            break

        loss_train.append(cur_train_loss)
        loss_test.append(cur_test_loss)

        # print progress
        print('Epoch: ', epoch,
              '| train loss: %.8f' % cur_train_loss,
              '| val loss: %.8f' % cur_test_loss,
              '| val f1: %.8f' % val_f1)
    if bool(args.show_grad_norm):
        times, data_prepare_times = times[int(200/args.batch_num):], data_prepare_times[int(200/args.batch_num):]
    print('Average training time is %0.3f' % np.mean(times))
    print('Average data prepare time is %0.3f'%np.mean(data_prepare_times))
    f1_score_test = best_model.calculate_f1(
        feat_data, adjs_full, labels, test_nodes)
    return best_model, loss_train, loss_test, loss_train_all, f1_score_test, grad_variance_all


"""
Variance Reduced Sampling GCN
"""

# def subgraph_gcn(feat_data, labels, adj_matrix, train_nodes, valid_nodes, test_nodes,  args, device, concat=False):
#     samp_num_list = np.array([args.samp_num for _ in range(args.n_layers)])
#     wrapper = ForwardWrapper(n_nodes=len(
#         feat_data), n_hid=args.nhid, n_layers=args.n_layers, n_classes=args.num_classes)

#     # use multiprocess sample data
#     process_ids = np.arange(args.batch_num)

#     subgraph_sampler_ = subgraph_sampler(adj_matrix, train_nodes)

#     if concat:
#         susage = GraphSageGCN(nfeat=feat_data.shape[1], nhid=args.nhid, num_classes=args.num_classes,
#                  layers=args.n_layers, dropout=args.dropout, multi_class=args.multi_class).to(device)
#     else:
#         susage = GCN(nfeat=feat_data.shape[1], nhid=args.nhid, num_classes=args.num_classes,
#                  layers=args.n_layers, dropout=args.dropout, multi_class=args.multi_class).to(device)

#     susage.to(device)
#     print(susage)

#     optimizer = optim.Adam(susage.parameters())

#     adjs_full, input_nodes_full, sampled_nodes_full = subgraph_sampler_.full_batch(
#         train_nodes, len(feat_data), args.n_layers)
#     adjs_full = package_mxl(adjs_full, device)

#     best_model = copy.deepcopy(susage)
#     susage.zero_grad()
#     cur_test_loss = susage.calculate_loss_grad(
#         feat_data, adjs_full, labels, valid_nodes)

#     best_val, cnt = 0, 0

#     loss_train = [cur_test_loss]
#     loss_test = [cur_test_loss]
#     grad_variance_all = []
#     loss_train_all = [cur_test_loss]
#     times = []
#     data_prepare_times = []
    
#     for epoch in np.arange(args.epoch_num):

#         train_nodes_p = args.batch_size * \
#             np.ones_like(train_nodes)/len(train_nodes)

#         susage.zero_grad()

#         # prepare train data
#         tp0 = time.time()
#         pool = mp.Pool(args.pool_num)
#         jobs = prepare_data(pool, subgraph_sampler_.mini_batch, process_ids, train_nodes, train_nodes_p, samp_num_list, len(feat_data),
#                             adj_matrix, args.n_layers)
#         # fetch train data
#         train_data = [job.get() for job in jobs]
#         pool.close()
#         pool.join()
#         tp1 = time.time()
#         data_prepare_times += [tp1-tp0]
        
#         inner_loop_num = args.batch_num

#         t2 = time.time()
#         cur_train_loss, cur_train_loss_all, grad_variance = variance_reduced_step(susage, optimizer, feat_data, labels,
#                                                                                   train_nodes, valid_nodes,
#                                                                                   adjs_full, train_data, inner_loop_num, device, wrapper,
#                                                                                   calculate_grad_vars=bool(args.show_grad_norm) if epoch<int(200/args.batch_num) else False)
#         t3 = time.time()

#         times += [t3-t2]
#         print('mvs_gcn_plus run time per epoch is %0.3f' % (t3-t2))

#         loss_train_all.extend(cur_train_loss_all)
#         grad_variance_all.extend(grad_variance)
#         # calculate test loss
#         susage.eval()

#         susage.zero_grad()
#         cur_test_loss = susage.calculate_loss_grad(
#             feat_data, adjs_full, labels, valid_nodes)
#         val_f1 = susage.calculate_f1(feat_data, adjs_full, labels, valid_nodes)

#         if val_f1 > best_val:
#             best_model = copy.deepcopy(susage)
#         if val_f1 > best_val + 1e-2:
#             best_val = val_f1
#             cnt = 0
#         else:
#             cnt += 1
#         if cnt == args.n_stops // args.batch_num:
#             break

#         loss_train.append(cur_train_loss)
#         loss_test.append(cur_test_loss)

#         # print progress
#         print('Epoch: ', epoch,
#               '| train loss: %.8f' % cur_train_loss,
#               '| val loss: %.8f' % cur_test_loss,
#               '| val f1: %.8f' % val_f1)
#     print('Average time is %0.3f' % np.mean(times))
#     print('Average data prepare time is %0.3f'%np.mean(data_prepare_times))
#     f1_score_test = best_model.calculate_f1(
#         feat_data, adjs_full, labels, test_nodes)
#     return best_model, loss_train, loss_test, loss_train_all, f1_score_test, grad_variance_all

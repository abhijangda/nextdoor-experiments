from samplers import vrgcn_sampler
from utils import *
from packages import *
from optimizers import sgd_step, variance_reduced_step
from forward_wrapper import ForwardWrapper
from model import GCN, GraphSageGCN
"""
GraphSage
"""


def graphsage(feat_data, labels, adj_matrix, train_nodes, valid_nodes, test_nodes,  args, device, concat=False):
    if args.dataset=='reddit':
        samp_num_list = np.array([25, 10])
    else:
        samp_num_list = np.array([5 for _ in range(args.n_layers)])
    
    # use multiprocess sample data
    process_ids = np.arange(args.batch_num)

    graphsage_sampler_ = graphsage_sampler(adj_matrix, train_nodes)

    if concat:
        susage = GraphSageGCN(nfeat=feat_data.shape[1], nhid=args.nhid, num_classes=args.num_classes,
                              layers=args.n_layers, dropout=args.dropout, multi_class=args.multi_class).to(device)
    else:
        susage = GCN(nfeat=feat_data.shape[1], nhid=args.nhid, num_classes=args.num_classes,
                     layers=args.n_layers, dropout=args.dropout, multi_class=args.multi_class).to(device)

    susage.to(device)
    print(susage)

    optimizer = optim.Adam(susage.parameters())

    adjs_full, input_nodes_full, sampled_nodes_full = graphsage_sampler_.full_batch(
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
        jobs = prepare_data(pool, graphsage_sampler_.mini_batch, process_ids, train_nodes, train_nodes_p, samp_num_list, len(feat_data),
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
    print('Average time is %0.3f' % np.mean(times))
    print('Average data prepare time is %0.3f'%np.mean(data_prepare_times))
    return best_model, loss_train, loss_test, loss_train_all, f1_score_test, grad_variance_all


"""
Exact inference
"""


# def exact_gcn(feat_data, labels, adj_matrix, train_nodes, valid_nodes, test_nodes,  args, device, concat=False):
#     samp_num_list = np.array([args.samp_num for _ in range(args.n_layers)])
#     # use multiprocess sample data
#     process_ids = np.arange(args.batch_num)

#     exact_sampler_ = exact_sampler(adj_matrix, train_nodes)

#     if concat:
#         susage = GraphSageGCN(nfeat=feat_data.shape[1], nhid=args.nhid, num_classes=args.num_classes,
#                               layers=args.n_layers, dropout=args.dropout, multi_class=args.multi_class).to(device)
#     else:
#         susage = GCN(nfeat=feat_data.shape[1], nhid=args.nhid, num_classes=args.num_classes,
#                      layers=args.n_layers, dropout=args.dropout, multi_class=args.multi_class).to(device)
#     susage.to(device)
#     print(susage)

#     optimizer = optim.Adam(susage.parameters())

#     adjs_full, input_nodes_full, sampled_nodes_full = exact_sampler_.full_batch(
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
#         # calculate gradients
#         susage.zero_grad()

#         train_nodes_p = args.batch_size * \
#             np.ones_like(train_nodes)/len(train_nodes)

#         # prepare train data
#         tp0 = time.time()
#         pool = mp.Pool(args.pool_num)
#         jobs = prepare_data(pool, exact_sampler_.mini_batch, process_ids, train_nodes, train_nodes_p, samp_num_list, len(feat_data),
#                             adj_matrix, args.n_layers)
#         # fetch train data
#         train_data = [job.get() for job in jobs]
#         pool.close()
#         pool.join()
#         tp1 = time.time()
#         data_prepare_times += [tp1-tp0]
        
#         inner_loop_num = args.batch_num

#         t2 = time.time()
#         cur_train_loss, cur_train_loss_all, grad_variance = sgd_step(susage, optimizer, feat_data, labels,
#                                                                      train_nodes, valid_nodes,
#                                                                      adjs_full, train_data, inner_loop_num, device,
#                                                                      calculate_grad_vars=bool(args.show_grad_norm) if epoch<int(200/args.batch_num) else False)
#         t3 = time.time()
#         times += [t3-t2]
#         print('mvs_gcn run time per epoch is %0.3f' % (t3-t2))

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

#     f1_score_test = best_model.calculate_f1(
#         feat_data, adjs_full, labels, test_nodes)
#     print('Average time is %0.3f' % np.mean(times))
#     print('Average data prepare time is %0.3f'%np.mean(data_prepare_times))
#     return best_model, loss_train, loss_test, loss_train_all, f1_score_test, grad_variance_all


"""
Variance Reduced GCN
"""


def vrgcn(feat_data, labels, adj_matrix, train_nodes, valid_nodes, test_nodes, args, device, concat=False):
    samp_num_list = np.array([2 for _ in range(args.n_layers)])
    wrapper = ForwardWrapper(n_nodes=len(
        feat_data), n_hid=args.nhid, n_layers=args.n_layers, n_classes=args.num_classes, concat=concat)

    vrgcn_sampler_ = vrgcn_sampler(adj_matrix, train_nodes)

    # use multiprocess sample data
    process_ids = np.arange(args.batch_num)

    if concat:
        susage = GraphSageGCN(nfeat=feat_data.shape[1], nhid=args.nhid, num_classes=args.num_classes,
                              layers=args.n_layers, dropout=args.dropout, multi_class=args.multi_class).to(device)
    else:
        susage = GCN(nfeat=feat_data.shape[1], nhid=args.nhid, num_classes=args.num_classes,
                     layers=args.n_layers, dropout=args.dropout, multi_class=args.multi_class).to(device)

    susage.to(device)
    print(susage)

    optimizer = optim.Adam(susage.parameters())

    adjs_full, input_nodes_full, sampled_nodes_full = vrgcn_sampler_.full_batch(
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
        jobs = prepare_data(pool, vrgcn_sampler_.mini_batch, process_ids, train_nodes, train_nodes_p, samp_num_list, len(feat_data),
                            adj_matrix, args.n_layers)
        # fetch train data
        train_data = [job.get() for job in jobs]
        pool.close()
        pool.join()
        tp1 = time.time()
        data_prepare_times += [tp1-tp0]

        inner_loop_num = args.batch_num

        t0 = time.time()
        cur_train_loss, cur_train_loss_all, grad_variance = variance_reduced_step(susage, optimizer, feat_data, labels,
                                                                                  train_nodes, valid_nodes,
                                                                                  adjs_full, train_data, inner_loop_num, device, wrapper,
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
    print('Average time is %0.3f' % np.mean(times))
    print('Average data prepare time is %0.3f'%np.mean(data_prepare_times))
    return best_model, loss_train, loss_test, loss_train_all, f1_score_test, grad_variance_all

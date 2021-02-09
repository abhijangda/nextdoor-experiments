from utils import *
from packages import *
from optimizers import sgd_step
from model import GCN, GraphSageGCN
"""
FastGCN
"""


# def fastgcn(feat_data, labels, adj_matrix, train_nodes, valid_nodes, test_nodes,  args, device):
#     samp_num_list = np.array([args.samp_num for _ in range(args.n_layers)])
#     # use multiprocess sample data
#     process_ids = np.arange(args.batch_num)

#     fastgcn_sampler_ = fastgcn_sampler(adj_matrix, train_nodes)
#     susage = GCN(nfeat=feat_data.shape[1], nhid=args.nhid, num_classes=args.num_classes,
#                  layers=args.n_layers, dropout=args.dropout, multi_class=args.multi_class).to(device)
#     susage.to(device)
#     print(susage)

#     optimizer = optim.Adam(
#         filter(lambda p: p.requires_grad, susage.parameters()))

#     adjs_full, input_nodes_full, sampled_nodes_full = fastgcn_sampler_.full_batch(
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

#         # prepare train data
#         tp0 = time.time()
#         pool = mp.Pool(args.pool_num)
#         jobs = prepare_data(pool, fastgcn_sampler_.mini_batch, process_ids, train_nodes, train_nodes_p, samp_num_list, len(feat_data),
#                             adj_matrix, args.n_layers)
#         # fetch train data
#         train_data = [job.get() for job in jobs]
#         pool.close()
#         pool.join()
#         tp1 = time.time()
#         data_prepare_times += [tp1-tp0]

#         inner_loop_num = args.batch_num

#         t0 = time.time()
#         cur_train_loss, cur_train_loss_all, grad_variance = sgd_step(susage, optimizer, feat_data, labels,
#                                                                      train_nodes, valid_nodes,
#                                                                      adjs_full, train_data, inner_loop_num, device,
#                                                                      calculate_grad_vars=bool(args.show_grad_norm) if epoch<int(200/args.batch_num) else False)
#         t1 = time.time()

#         times += [t1-t0]
#         print('sgcn run time per epoch is %0.3f' % (t1-t0))
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
#               '| test loss: %.8f' % cur_test_loss,
#               '| test f1: %.8f' % val_f1)

#     f1_score_test = best_model.calculate_f1(
#         feat_data, adjs_full, labels, test_nodes)
#     print('Average training time is %0.3f' % np.mean(times))
#     print('Average data prepare time is %0.3f'%np.mean(data_prepare_times))
#     return best_model, loss_train, loss_test, loss_train_all, f1_score_test, grad_variance_all


def ladies(feat_data, labels, adj_matrix, train_nodes, valid_nodes, test_nodes,  args, device, concat=True):
    samp_num_list = np.array([args.samp_num for _ in range(args.n_layers)])
    # use multiprocess sample data
    process_ids = np.arange(args.batch_num)

    ladies_sampler_ = ladies_sampler(adj_matrix, train_nodes)
    
    if concat:        
        susage = GraphSageGCN(nfeat=feat_data.shape[1], nhid=args.nhid, num_classes=args.num_classes,
                     layers=args.n_layers, dropout=args.dropout, multi_class=args.multi_class).to(device)
    else:
        susage = GCN(nfeat=feat_data.shape[1], nhid=args.nhid, num_classes=args.num_classes,
                     layers=args.n_layers, dropout=args.dropout, multi_class=args.multi_class).to(device)
    
    susage.to(device)
    print(susage)

    optimizer = optim.Adam(susage.parameters())

    adjs_full, input_nodes_full, sampled_nodes_full = ladies_sampler_.full_batch(
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
        if concat:
            jobs = prepare_data(pool, ladies_sampler_.mini_batch_ld, process_ids, train_nodes, train_nodes_p, samp_num_list, len(feat_data),
                            adj_matrix, args.n_layers) 
        else:
            jobs = prepare_data(pool, ladies_sampler_.mini_batch, process_ids, train_nodes, train_nodes_p, samp_num_list, len(feat_data),
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

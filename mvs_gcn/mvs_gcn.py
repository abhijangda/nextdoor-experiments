from utils import *
from packages import *
import autograd_wl
from optimizers import boost_step, variance_reduced_boost_step
from forward_wrapper import ForwardWrapper_v2 as ForwardWrapper
from model import GCN, GraphSageGCN

"""
Minimal Variance Sampling GCN
"""


# def mvs_gcn(feat_data, labels, adj_matrix, train_nodes, valid_nodes, test_nodes,  args, device, concat=False):
#     samp_num_list = np.array([args.samp_num for _ in range(args.n_layers)])
#     # use multiprocess sample data
#     process_ids = np.arange(args.batch_num)

#     exact_sampler_ = exact_sampler(adj_matrix, train_nodes)
#     if concat:
#         susage = GraphSageGCN(nfeat=feat_data.shape[1], nhid=args.nhid, num_classes=args.num_classes,
#                           layers=args.n_layers, dropout=args.dropout, multi_class=args.multi_class).to(device)
#     else:
#         susage = GCN(nfeat=feat_data.shape[1], nhid=args.nhid, num_classes=args.num_classes,
#                           layers=args.n_layers, dropout=args.dropout, multi_class=args.multi_class).to(device)

#     susage.to(device)
#     print(susage)

#     optimizer = optim.Adam(susage.parameters())

#     adjs_full, input_nodes_full, sampled_nodes_full = exact_sampler_.full_batch(
#             train_nodes, len(feat_data), args.n_layers)
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
#     full_batch_times = []
#     data_prepare_times = []
    
#     for epoch in np.arange(args.epoch_num):
#         # calculate gradients
#         susage.zero_grad()

#         mini_batch_nodes = np.random.permutation(
#             len(train_nodes))[:int(len(train_nodes)*args.is_ratio)]
#         grad_per_sample = np.zeros_like(train_nodes, dtype=np.float32)
#         adjs_mini, input_nodes_mini, sampled_nodes_mini = exact_sampler_.large_batch(
#             train_nodes[mini_batch_nodes], len(feat_data), args.n_layers)
#         adjs_mini = package_mxl(adjs_mini, device)

#         t0 = time.time()
#         grad_per_sample[mini_batch_nodes] = susage.calculate_sample_grad(
#             feat_data[input_nodes_mini], adjs_mini, labels, train_nodes[mini_batch_nodes])
#         t1 = time.time()
#         full_batch_times += [t1-t0]
        
#         thresh = CalculateThreshold(grad_per_sample, args.batch_size)
#         train_nodes_p = grad_per_sample/thresh
#         train_nodes_p[train_nodes_p > 1] = 1

#         # prepare train data
#         tp_0 = time.time()
#         pool = mp.Pool(args.pool_num)
#         jobs = prepare_data(pool, exact_sampler_.mini_batch, process_ids, train_nodes, train_nodes_p, samp_num_list, len(feat_data),
#                             adj_matrix, args.n_layers, args.is_ratio)
#         # fetch train data
#         train_data = [job.get() for job in jobs]
#         pool.close()
#         pool.join()
#         tp_1 = time.time()
#         data_prepare_times += [tp_1-tp_0]
        
#         inner_loop_num = args.batch_num

#         t2 = time.time()
#         cur_train_loss, cur_train_loss_all, grad_variance = boost_step(susage, optimizer, feat_data, labels,
#                                           train_nodes, valid_nodes,
#                                           adjs_full, train_data, inner_loop_num, device,
#                                           calculate_grad_vars=bool(args.show_grad_norm) if epoch<int(200/args.batch_num) else False)
#         t3 = time.time()
#         times += [t3-t2]
#         print('mvs_gcn run time per epoch is %0.3f' % (t1-t0+t3-t2))

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
#     print('Average training time is %0.3f'%np.mean(times))
#     print('Average full batch time is %0.3f'%np.mean(full_batch_times))
#     print('Average data prepare time is %0.3f'%np.mean(data_prepare_times))
#     return best_model, loss_train, loss_test, loss_train_all, f1_score_test, grad_variance_all

# """
# minimal variance sampling with online learning (on the fly)
# """
# def mvs_gcn_otf(feat_data, labels, adj_matrix, train_nodes, valid_nodes, test_nodes,  args, device, concat=False):
#     from optimizers import calculate_grad_variance
    
#     samp_num_list = np.array([args.samp_num for _ in range(args.n_layers)])
#     process_ids = np.arange(args.batch_num)

#     exact_sampler_ = exact_sampler(adj_matrix, train_nodes)
#     if concat:
#         susage = GraphSageGCN(nfeat=feat_data.shape[1], nhid=args.nhid, num_classes=args.num_classes,
#                           layers=args.n_layers, dropout=args.dropout, multi_class=args.multi_class).to(device)
#     else:
#         susage = GCN(nfeat=feat_data.shape[1], nhid=args.nhid, num_classes=args.num_classes,
#                           layers=args.n_layers, dropout=args.dropout, multi_class=args.multi_class).to(device)

#     susage.to(device)
#     print(susage)

#     optimizer = optim.Adam(susage.parameters())

#     adjs_full, input_nodes_full, sampled_nodes_full = exact_sampler_.full_batch(
#             train_nodes, len(feat_data), args.n_layers)
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
    
#     # before started, every nodes has the save probability
#     sample_ratio = args.batch_size/len(train_nodes)
#     train_nodes_p = np.ones_like(train_nodes, dtype=np.float32) * sample_ratio
    
#     for epoch in np.arange(args.epoch_num):
        
#         #######################
#         #######################
#         #######################
#         susage.train()
#         cur_train_loss_all = []
#         grad_variance = []
        
#         t2 = time.time()
#         for iter_num in range(args.batch_num):
#             sample_mask = np.random.uniform(0, 1, len(train_nodes))<= train_nodes_p
#             probs_nodes = train_nodes_p[sample_mask] * len(train_nodes)
#             batch_nodes = train_nodes[sample_mask]
#             adjs, input_nodes, output_nodes, probs_nodes, sampled_nodes = \
#                 exact_sampler_.mini_batch(iter_num, batch_nodes, probs_nodes, samp_num_list, len(feat_data), adj_matrix, args.n_layers)
#             adjs = package_mxl(adjs, device)
            
#             optimizer.zero_grad()
#             weight = 1.0/torch.FloatTensor(probs_nodes).to(device)
#             current_loss, current_grad_norm = susage.partial_grad_with_norm(
#                 feat_data[input_nodes], adjs, labels[output_nodes], weight)
            
#             # only for experiment purpose to demonstrate ...
#             if bool(args.show_grad_norm):
#                 grad_variance.append(calculate_grad_variance(
#                     susage, feat_data, labels, train_nodes, adjs_full))

#             optimizer.step()
            
#             # print statistics
#             cur_train_loss_all += [current_loss.cpu().detach()]
            
#             # update train_nodes_p
#             thresh = CalculateThreshold(current_grad_norm, args.batch_size*sample_ratio)
#             current_node_p= current_grad_norm/thresh
#             current_node_p[current_node_p > 1] = 1
#             train_nodes_p[sample_mask] = current_node_p
            
#         t3 = time.time()
#         # calculate training loss
#         cur_train_loss = np.mean(cur_train_loss_all)
#         #######################
#         #######################
#         #######################

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
#     print('Average training time is %0.3f'%np.mean(times))
#     return best_model, loss_train, loss_test, loss_train_all, f1_score_test, grad_variance_all


"""
Minimal Variance Sampling GCN +
"""
def mvs_gcn_plus(feat_data, labels, adj_matrix, train_nodes, valid_nodes, test_nodes,  args, device, concat=False, fq=0, increase=False):
    samp_num_list = np.array([args.samp_num for _ in range(args.n_layers)])
    wrapper = ForwardWrapper(n_nodes=len(feat_data), n_hid=args.nhid, n_layers=args.n_layers, n_classes=args.num_classes, concat=concat)
    
    # use multiprocess sample data
    process_ids = np.arange(args.batch_num)

    subgraph_sampler_ = subgraph_sampler(adj_matrix, train_nodes)

    if concat:
        susage = GraphSageGCN(nfeat=feat_data.shape[1], nhid=args.nhid, num_classes=args.num_classes,
                 layers=args.n_layers, dropout=args.dropout, multi_class=args.multi_class).to(device)
    else:
        susage = GCN(nfeat=feat_data.shape[1], nhid=args.nhid, num_classes=args.num_classes,
                 layers=args.n_layers, dropout=args.dropout, multi_class=args.multi_class).to(device)
    
    susage.to(device)
    print(susage)
        
    optimizer = optim.Adam(susage.parameters())

    adjs_full, input_nodes_full, sampled_nodes_full = subgraph_sampler_.full_batch(
            train_nodes, len(feat_data), args.n_layers)
    adjs_full = package_mxl(adjs_full, device)

    best_model = copy.deepcopy(susage)
    susage.zero_grad()
    cur_test_loss = susage.calculate_loss_grad(feat_data, adjs_full, labels, valid_nodes)
        
    best_val, cnt = 0, 0

    loss_train = [cur_test_loss]
    loss_test = [cur_test_loss]
    grad_variance_all = []
    loss_train_all = [cur_test_loss]
    times = []
    full_batch_times = []
    data_prepare_times = []
    
    epoch_cnt = 0
    for epoch in np.arange(args.epoch_num):
        
        susage.zero_grad()
        
        if epoch%(fq/args.batch_num)==0:
            print(fq/args.batch_num)
            if increase:
                fq += 10
            mini_batch_nodes = np.random.permutation(len(train_nodes))[:int(len(train_nodes)*args.is_ratio)]
    
            grad_per_sample = np.zeros_like(train_nodes, dtype=np.float32)
            adjs_mini, input_nodes_mini, sampled_nodes_mini = subgraph_sampler_.large_batch(
                train_nodes[mini_batch_nodes], len(feat_data), args.n_layers)
            adjs_mini = package_mxl(adjs_mini, device)
            
            t0 = time.time()
            # optimizer.zero_grad()
            grad_per_sample[mini_batch_nodes] = wrapper.calculate_sample_grad(
                susage, feat_data[input_nodes_mini], adjs_mini, sampled_nodes_mini, labels, train_nodes[mini_batch_nodes])
            # optimizer.step()
            t1 = time.time()

            full_batch_times += [t1-t0]
    
            thresh = CalculateThreshold(grad_per_sample, args.batch_size)
            train_nodes_p = grad_per_sample/thresh
            train_nodes_p[train_nodes_p>1] = 1
            
        # prepare train data
        tp0 = time.time()
        asyncSampling = False
        if asyncSampling:
            pool = mp.Pool(args.pool_num)
            jobs = prepare_data(pool, subgraph_sampler_.mini_batch, process_ids, train_nodes, train_nodes_p, samp_num_list, len(feat_data),
                            adj_matrix, args.n_layers, args.is_ratio)
        # fetch train data
            train_data = [job.get() for job in jobs]
            pool.close()
            pool.join()
        else:
            num_train_nodes = len(train_nodes)
            num_nodes = len(feat_data)
            is_ratio = args.is_ratio
            sample_mask = np.random.uniform(0, 1, num_train_nodes) <= train_nodes_p
            probs_nodes = train_nodes_p[sample_mask] * len(train_nodes) * is_ratio
            batch_nodes = train_nodes[sample_mask]
            print(num_train_nodes, len(batch_nodes))
            train_data = []
            _t0 = time.time()
            for p in process_ids:
                train_data += [subgraph_sampler_.mini_batch(np.random.randint(2**32 - 1), batch_nodes, probs_nodes,
                            samp_num_list, num_nodes, adj_matrix, args.n_layers)]
            _t1 = time.time()


        tp1 = time.time()
        print("353:", tp1-tp0, _t1-_t0)
        data_prepare_times += [tp1-tp0]
        
        inner_loop_num = args.batch_num
        training_t1 = time.time()
        t2 = time.time()
        cur_train_loss, cur_train_loss_all, grad_variance = variance_reduced_boost_step(susage, optimizer, feat_data, labels,
                                          train_nodes, valid_nodes,
                                          adjs_full, train_data, inner_loop_num, device, wrapper,
                                          calculate_grad_vars=bool(args.show_grad_norm) if epoch<int(200/args.batch_num) else False)
        t3 = time.time()
        
        times += [t3-t2]
        print('mvs_gcn_plus run time per epoch is %0.3f'%(t1-t0 + t3-t2))


        loss_train_all.extend(cur_train_loss_all)
        grad_variance_all.extend(grad_variance)
        # calculate test loss
        susage.eval()

        susage.zero_grad()
        cur_test_loss = susage.calculate_loss_grad(
            feat_data, adjs_full, labels, valid_nodes)
        val_f1 = susage.calculate_f1(feat_data, adjs_full, labels, valid_nodes)
        training_t2 = time.time()
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
              '| val f1: %.8f' % val_f1,
              '| sampling time: %.8f' % (tp1-tp0),
              '| training time: %.8f' % (training_t2 - training_t1),
              '| variance reduced boost step time: %.8f' %(t3-t2))
    if False and bool(args.show_grad_norm):
        times, full_batch_times, data_prepare_times = \
            times[int(200/args.batch_num):], full_batch_times[int(200/args.batch_num):], data_prepare_times[int(200/args.batch_num):]
    print('Average training time is %0.3f'%np.mean(times))
    print('Average full batch time is %0.3f'%np.mean(full_batch_times))
    print('Average data prepare time is %0.3f'%np.mean(data_prepare_times))
    f1_score_test = best_model.calculate_f1(feat_data, adjs_full, labels, test_nodes)
    return best_model, loss_train, loss_test, loss_train_all, f1_score_test, grad_variance_all

"""
minimal variance sampling with online learning and subgraph sampling (on the fly)
"""
def mvs_gcn_plus_otf(feat_data, labels, adj_matrix, train_nodes, valid_nodes, test_nodes,  args, device, concat=False, fq=0, increase=False):
    from optimizers import calculate_grad_variance
    wrapper = ForwardWrapper(n_nodes=len(feat_data), n_hid=args.nhid, n_layers=args.n_layers, n_classes=args.num_classes, concat=concat)
    
    samp_num_list = np.array([args.samp_num for _ in range(args.n_layers)])
    process_ids = np.arange(args.batch_num)

    subgraph_sampler_ = subgraph_sampler(adj_matrix, train_nodes)
    if concat:
        susage = GraphSageGCN(nfeat=feat_data.shape[1], nhid=args.nhid, num_classes=args.num_classes,
                          layers=args.n_layers, dropout=args.dropout, multi_class=args.multi_class).to(device)
    else:
        susage = GCN(nfeat=feat_data.shape[1], nhid=args.nhid, num_classes=args.num_classes,
                          layers=args.n_layers, dropout=args.dropout, multi_class=args.multi_class).to(device)

    susage.to(device)
    print(susage)

    optimizer = optim.Adam(susage.parameters())

    adjs_full, input_nodes_full, sampled_nodes_full = subgraph_sampler_.full_batch(
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
    full_batch_times = []
    
    for epoch in np.arange(args.epoch_num):
        if epoch%(fq/args.batch_num)==0:
            print(fq/args.batch_num)
            if increase:
                fq += 10
            mini_batch_nodes = np.random.permutation(len(train_nodes))[:int(len(train_nodes)*args.is_ratio)]
    
            adjs_mini, input_nodes_mini, sampled_nodes_mini = subgraph_sampler_.large_batch(
                train_nodes[mini_batch_nodes], len(feat_data), args.n_layers)
            adjs_mini = package_mxl(adjs_mini, device)
            
            t0 = time.time()
            optimizer.zero_grad()
            grad_per_sample_in_mini_batch = wrapper.calculate_sample_grad(
                susage, feat_data[input_nodes_mini], adjs_mini, sampled_nodes_mini, labels, train_nodes[mini_batch_nodes])
            optimizer.step()
            t1 = time.time()

            full_batch_times += [t1-t0]
            grad_per_sample = np.zeros_like(train_nodes, dtype=np.float32)
            grad_per_sample[mini_batch_nodes] = grad_per_sample_in_mini_batch
            
        #######################
        #######################
        #######################
        susage.train()
        cur_train_loss_all = []
        grad_variance = []
        
        times_ = []
        times_tp_ = []
        for iter_num in range(args.batch_num):
            thresh = CalculateThreshold(grad_per_sample, args.batch_size)
            train_nodes_p = grad_per_sample/thresh
            train_nodes_p[train_nodes_p>1] = 1
            
            sample_mask = np.random.uniform(0, 1, len(train_nodes))<= train_nodes_p
            probs_nodes = train_nodes_p[sample_mask] * len(train_nodes) * args.is_ratio
            batch_nodes = train_nodes[sample_mask]

            t0 = time.time()
            adjs, adjs_exact, input_nodes, output_nodes, probs_nodes, sampled_nodes, input_exact_nodes = \
                subgraph_sampler_.mini_batch(iter_num, batch_nodes, probs_nodes, samp_num_list, len(feat_data), adj_matrix, args.n_layers)
            t1 = time.time()
            times_tp_ += [t1-t0]
            
            t2 = time.time()
            adjs, adjs_exact = package_mxl(adjs, device), package_mxl(adjs_exact, device)
            optimizer.zero_grad()
            weight = 1.0/torch.FloatTensor(probs_nodes).to(device)
            current_loss, current_grad_norm = wrapper.partial_grad_with_norm(susage, 
                feat_data[input_nodes], adjs, sampled_nodes, feat_data, adjs_exact, input_exact_nodes, labels[output_nodes], weight)
            t3 = time.time()
            times_ += [t3-t2]
            # only for experiment purpose to demonstrate ...
            if bool(args.show_grad_norm) if epoch<int(200/args.batch_num) else False:
                grad_variance.append(calculate_grad_variance(
                    susage, feat_data, labels, train_nodes, adjs_full))

            optimizer.step()
            
            # print statistics
            cur_train_loss_all += [current_loss.cpu().detach()]
            
            # update train_nodes_p
            grad_per_sample[sample_mask] = current_grad_norm

        # calculate training loss
        cur_train_loss = np.mean(cur_train_loss_all)
        #######################
        #######################
        #######################

        times += [sum(times_)]
        print('mvs_gcn run time per epoch is %0.3f' % sum(times_))

        data_prepare_times += [sum(times_tp_)]

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

    f1_score_test = best_model.calculate_f1(
        feat_data, adjs_full, labels, test_nodes)
    
    if bool(args.show_grad_norm):
        times, full_batch_times, data_prepare_times = \
            times[int(200/args.batch_num):], full_batch_times[int(200/args.batch_num):], data_prepare_times[int(200/args.batch_num):]
    print('Average training time is %0.3f'%np.mean(times))
    print('Average full batch time is %0.3f'%np.mean(full_batch_times))
    print('Average data prepare time is %0.3f'%np.mean(data_prepare_times))
    return best_model, loss_train, loss_test, loss_train_all, f1_score_test, grad_variance_all

# def mvs_gcn_plus_otf(feat_data, labels, adj_matrix, train_nodes, valid_nodes, test_nodes,  args, device, concat=False):
#     from optimizers import calculate_grad_variance
#     wrapper = ForwardWrapper(n_nodes=len(feat_data), n_hid=args.nhid, n_layers=args.n_layers, n_classes=args.num_classes, concat=concat)
    
#     samp_num_list = np.array([args.samp_num for _ in range(args.n_layers)])
#     process_ids = np.arange(args.batch_num)

#     subgraph_sampler_ = subgraph_sampler(adj_matrix, train_nodes)
#     if concat:
#         susage = GraphSageGCN(nfeat=feat_data.shape[1], nhid=args.nhid, num_classes=args.num_classes,
#                           layers=args.n_layers, dropout=args.dropout, multi_class=args.multi_class).to(device)
#     else:
#         susage = GCN(nfeat=feat_data.shape[1], nhid=args.nhid, num_classes=args.num_classes,
#                           layers=args.n_layers, dropout=args.dropout, multi_class=args.multi_class).to(device)

#     susage.to(device)
#     print(susage)

#     optimizer = optim.Adam(susage.parameters())

#     adjs_full, input_nodes_full, sampled_nodes_full = subgraph_sampler_.full_batch(
#             train_nodes, len(feat_data), args.n_layers)
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
#     full_batch_times = []
#     # before started, every nodes has the save probability
#     sample_ratio = args.batch_size/len(train_nodes)
#     train_nodes_p = np.ones_like(train_nodes, dtype=np.float32) * sample_ratio
    
#     # warm start
#     wrapper.forward_full(susage, feat_data, adjs_full, sampled_nodes_full)
    
#     for epoch in np.arange(args.epoch_num):
#         tp0 = time.time()
#         wrapper.forward_full(susage, feat_data, adjs_full, sampled_nodes_full)
#         tp1 = time.time()
#         full_batch_times += [tp1-tp0]
#         #######################
#         #######################
#         #######################
#         susage.train()
#         cur_train_loss_all = []
#         grad_variance = []
        
#         times_ = []
#         times_tp_ = []
#         for iter_num in range(args.batch_num):
#             sample_mask = np.random.uniform(0, 1, len(train_nodes))<= train_nodes_p
#             probs_nodes = train_nodes_p[sample_mask] * len(train_nodes)
#             batch_nodes = train_nodes[sample_mask]

#             t0 = time.time()
#             adjs, adjs_exact, input_nodes, output_nodes, probs_nodes, sampled_nodes, input_exact_nodes = \
#                 subgraph_sampler_.mini_batch(iter_num, batch_nodes, probs_nodes, samp_num_list, len(feat_data), adj_matrix, args.n_layers)
#             t1 = time.time()
#             times_tp_ += [t1-t0]
            
#             t2 = time.time()
#             adjs, adjs_exact = package_mxl(adjs, device), package_mxl(adjs_exact, device)
#             optimizer.zero_grad()
#             weight = 1.0/torch.FloatTensor(probs_nodes).to(device)
#             current_loss, current_grad_norm = wrapper.partial_grad_with_norm(susage, 
#                 feat_data[input_nodes], adjs, sampled_nodes, feat_data, adjs_exact, input_exact_nodes, labels[output_nodes], weight)
#             t3 = time.time()
#             times_ += [t3-t2]
#             # only for experiment purpose to demonstrate ...
#             if bool(args.show_grad_norm) if epoch<int(200/args.batch_num) else False:
#                 grad_variance.append(calculate_grad_variance(
#                     susage, feat_data, labels, train_nodes, adjs_full))

#             optimizer.step()
            
#             # print statistics
#             cur_train_loss_all += [current_loss.cpu().detach()]
            
#             # update train_nodes_p
#             thresh = CalculateThreshold(current_grad_norm, args.batch_size*sample_ratio)
#             current_node_p= current_grad_norm/thresh
#             current_node_p[current_node_p > 1] = 1
#             train_nodes_p[sample_mask] = current_node_p
            
#         # calculate training loss
#         cur_train_loss = np.mean(cur_train_loss_all)
#         #######################
#         #######################
#         #######################

#         times += [sum(times_)]
#         print('mvs_gcn run time per epoch is %0.3f' % sum(times_))

#         data_prepare_times += [sum(times_tp_)]

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
    
#     if bool(args.show_grad_norm):
#         times, full_batch_times, data_prepare_times = \
#             times[int(200/args.batch_num):], full_batch_times[int(200/args.batch_num):], data_prepare_times[int(200/args.batch_num):]
#     print('Average training time is %0.3f'%np.mean(times))
#     print('Average full batch time is %0.3f'%np.mean(full_batch_times))
#     print('Average data prepare time is %0.3f'%np.mean(data_prepare_times))
#     return best_model, loss_train, loss_test, loss_train_all, f1_score_test, grad_variance_all
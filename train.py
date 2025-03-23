import os
from random import sample
import dill
import torch
import numpy as np
import sys
import json
import math
from torch.nn import init
from tqdm import tqdm
import time
import pickle
import argparse
from pathlib import Path
from methods import get_model
import copy
from utils.data_processing import get_data, computer_time_statics,get_data_old
from utils.utils import *
from utils.evaluation import eval_prediction
from utils.log_and_checkpoints import *
from models.Backbone import TemporalGNNClassifier
import matplotlib.pyplot as plt
import seaborn
from copy import deepcopy
from distutils.util import strtobool
import wandb 
import warnings
parser = argparse.ArgumentParser('TGCL')

# training settings


parser.add_argument('--verbose', type=int, default=0, help='debug mode')
parser.add_argument('--seed', type=int, default=5, help='seed')
parser.add_argument('--device', type=int, default=0, help='Device of cuda')
parser.add_argument('--filename_add', type=str, default='', help='Attachment to filename')

# general parameters
parser.add_argument('--normalized_timestamps', type=bool, default=True, help='Normalize timestamps')
parser.add_argument('--rp_times', type=int, default=1, help='repeat running times')
parser.add_argument('--dataset', type=str,default='yelp_clear')
parser.add_argument('--model', type=str, default='TGAT', help='Model')
parser.add_argument('--method', type=str, default='NForGOT', help='Continual learning method')
parser.add_argument('--fuzzy_boundary', type=int, default=1, help='if fuzzy boundary')
parser.add_argument('--average_stopper', type=int, default=0, help='if use average stopper')
parser.add_argument('--stop_method', type=str, default='acc', help='early stop method')
parser.add_argument('--stop_tol', type=float, default=1e-10, help='early stop tolerance')
# parser.add_argument('--model', type=str, default='OTGNet', help='Model')
parser.add_argument('--batch_size', type=int, default=300, help='Batch_size')
parser.add_argument('--n_epoch', type=int, default=100, help='Number of epochs')
parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
parser.add_argument('--select', type=str, default='reinforce', help='Policy select')

# parser.add_argument('--n_interval', type=int, default=3, help='Interval of RL training')
# parser.add_argument('--n_mc', type=int, default=3, help='Number of MC Dropout')
# parser.add_argument('--num_class', type=int, default=3, help='Number of classes')
parser.add_argument('--num_class_per_dataset', type=int, default=3, help='Number of classes per dataset')
parser.add_argument('--num_datasets', type=int, default=8, help='Number of datasets')
parser.add_argument('--num_neighbors', type=int, default=5, help='Number of neighbors to sample') 

## Backbone model parameters
parser.add_argument('--supervision', type=str, default='supervised', help='Supervision type')
parser.add_argument('--task', type=str, default='nodecls', help='Task type')
parser.add_argument('--feature_type', type=str, default='both', help='The type of features used for node classification')
parser.add_argument('--time_dim', type=int, default=100, help='dimension of the time embedding')

parser.add_argument('--multihead', type=int, default=1, help='whether to use multihead classifiers for each data set')
parser.add_argument('--head_hidden_dim', type=int, default=100, help='Number of hidden dimensions of the head classifier')
parser.add_argument('--num_layer', type=int, default=1, help='Number of TGNN layers')
parser.add_argument('--dropout', type=float, default=0.3, help='Dropout rate')   

parser.add_argument('--sample_neighbor_strategy', type=str, default='recent', choices=['uniform', 'recent', 'time_interval_aware'], help='how to sample historical neighbors')
parser.add_argument('--time_scaling_factor', default=1e-6, type=float, help='the hyperparameter that controls the sampling preference with time interval, '
                        'a large time_scaling_factor tends to sample more on recent links, 0.0 corresponds to uniform sampling, '
                        'it works when sample_neighbor_strategy == time_interval_aware')
parser.add_argument('--patch_size', type=int, default=1, help='patch size')
parser.add_argument('--channel_embedding_dim', type=int, default=50, help='dimension of each channel embedding')
parser.add_argument('--max_input_sequence_length', type=int, default=32, help='maximal length of the input sequence of each node')
# continual learning method parameters
## Ours parameters
parser.add_argument('--linear_mmd', type=int, default=1, help='Use memory buffer or not')
parser.add_argument('--class_loss', type=float, default=0.01, help='Use memory buffer or not')
parser.add_argument('--alpha', type=float, default=1.0) 
parser.add_argument('--beta', type=float, default=1.0)                  
parser.add_argument('--distill_loss', type=float, default=1.0) 
parser.add_argument('--T', type=float, default=2.0)                 
parser.add_argument('--mmd_loss', type=float, default = 1.0)                 

parser.add_argument('--ewc_args', type=int, default=100000) # memory_strength
parser.add_argument('--gem_args', type=str2dict, default={'memory_strength': 0.5, 'n_memories': 100})
# Memory parameters

parser.add_argument('--memory_replay', type=int, default=0, help='Use memory buffer or not')
parser.add_argument('--select_mode', type=str, default='random', help='How to select the data into the memory')
parser.add_argument('--memory_size', type=float, default=100, help='Size of memory buffer')
parser.add_argument('--memory_frac', type=float, default=-1, help='Size of memory buffer')
parser.add_argument('--memory_replay_weight', type=int, default=1, help='Weight for replaying memory')
parser.add_argument('--replay_select_mode', type=str, default='random', help='How to select the data from the memory')
parser.add_argument('--replay_size', type=int, default=100, help='The number of data to replay')

parser.add_argument('--explainer', type=str, default='PGExplainer', help='Explainer')
parser.add_argument('--explainer_train_epoch', type=int, default=100, help='Number of epochs to train the explainer')
parser.add_argument('--explainer_lr', type=float, default=0.001, help='Learning rate of the explainer')
parser.add_argument('--explainer_batch_size', type=int, default=100, help='Batch size of the explainer')
parser.add_argument('--explainer_reg_coefs', type=float, default=0.1, help='Regularization coefficient of the explainer')
parser.add_argument('--explainer_level', type=str, default='node', help='the explanation level, node or graph')

# backbone model parameters

parser.add_argument('--use_feature', type=str, default='fg', help='Use node feature or not')
parser.add_argument('--use_time', type=int, default=5, help='Use time or not')
parser.add_argument('--mem_method', type=str, default='triad', help='Memory buffer sample method')
parser.add_argument('--ergnn_sampler', type=str, default='CM', help='sampler for ergnn')

# parser.add_argument('--ergnn_budget', type=int, default=0, help='Use memory buffer or not')
parser.add_argument('--mem_size', type=int, default=10, help='Size of memory slots')

parser.add_argument('--is_r', type=int, default=1, help='is_r')
parser.add_argument('--blurry', type=int, default=1, help='blurry setting')
parser.add_argument('--online', type=int, default=0, help='online setting')

## otgnet
parser.add_argument('--use_IB', type=int, default=1, help='use IB')
parser.add_argument('--dis_IB', type=int, default=1, help='dis IB')
parser.add_argument('--ch_IB', type=str, default='m', help='ch IB')
parser.add_argument('--pattern_rho', type=float, default=0.1, help='pattern_rho')
parser.add_argument('--num_attn_heads', type=int, default=2, help='Number of attention heads')

parser.add_argument('--node_init_dim', type=int, default=128, help='node initial feature dimension')
parser.add_argument('--node_embedding_dim', type=int, default=128, help='node embedding feature dimension')


parser.add_argument('--feature_iter', type=int, default=1, help='feature_iter')
parser.add_argument('--patience', type=int, default=20, help='patience')
parser.add_argument('--radius', type=float, default=0, help='radius')
# parser.add_argument('--beta', type=float, default=0, help='beta')
parser.add_argument('--gamma', type=float, default=0, help='gamma')
parser.add_argument('--uml', type=int, default=0, help='uml')
parser.add_argument('--pmethod', type=str, default='knn', help='pseudo-label method')
parser.add_argument('--sk', type=int, default=1000, help='number of triads candidates')
parser.add_argument('--full_n', type=int, default=1, help='full_n')
parser.add_argument('--recover', type=int, default=1, help='recover')

# training setting

parser.add_argument('--class_balance', type=int, default=1, help='class balance')
parser.add_argument('--eval_avg', type=str, default='node', help='evaluation average')

parser.add_argument('--results_dir', type=str, default='.', help='results diretion')
parser.add_argument('--explainer_ckpt_dir', type=str, default='.', help='check point direction for the explainer')


log_to_file = True
args = parser.parse_args()
if args.dataset == 'reddit':
    args.num_datasets = 6
    args.num_class_per_dataset = 3
elif args.dataset == 'yelp':
    args.num_datasets = 8
    args.num_class_per_dataset = 3
elif args.dataset == 'taobao':
    args.num_datasets = 3
    args.num_class_per_dataset = 30
elif args.dataset == 'amazon':
    args.num_datasets = 7
    args.num_class_per_dataset = 2
elif args.dataset == 'yelp_clear':
    args.num_datasets = 6
    args.num_class_per_dataset = 2

args.num_class = args.num_datasets * args.num_class_per_dataset


args.memory_replay = args.memory_replay==1
args.multihead = args.multihead==1
use_feature = args.use_feature
use_time = args.use_time
blurry = args.blurry==1
# online = args.online==1
is_r = args.is_r==1
mem_method = args.mem_method
mem_size = args.mem_size
rp_times = args.rp_times

use_IB = args.use_IB==1
dis_IB = args.dis_IB==1
ch_IB = args.ch_IB
pattern_rho = args.pattern_rho
class_balance = args.class_balance
eval_avg = args.eval_avg
feature_iter=args.feature_iter==1
patience=args.patience
radius = args.radius
beta = args.beta
gamma = args.gamma
uml = args.uml==1

pmethod = args.pmethod
sk = args.sk
full_n = args.full_n==1
recover = args.recover==1

avg_performance_all = []
avg_forgetting_all = []
forgetting_max_all = []
only_forgetting_all = []
bwt_all = []
overall_time = 0

device = args.device
torch.cuda.set_device(device)



task_acc_vary = [[[0,0,0,0] for _ in range(args.num_datasets)] for _ in range(args.num_datasets)]
# task_acc_vary_cur=[[0]*args.num_datasets for i in range(args.num_datasets)]
task_acc_vary_cur = [[[0,0,0,0] for _ in range(args.num_datasets)] for _ in range(args.num_datasets)]
val_test =  [[[0,0,0,0] for _ in range(args.num_datasets)] for _ in range(args.num_datasets)]
metrix = ['n_acc','acc','ap','f1']

# data processing
# if args.dataset == 'yelp':
#     node_features, edge_features, full_data, train_data, val_data, test_data, all_data, re_train_data, re_val_data = get_data(args.dataset,args.num_datasets,args.num_class_per_dataset,blurry,args.fuzzy_boundary)
# else: 
node_features, edge_features, full_data, train_data, val_data, test_data, all_data, re_train_data, re_val_data = get_data_old(args.dataset,args.num_datasets,args.num_class_per_dataset,blurry)
# if args.normalized_timestamps:
#     for k in range(args.num_datasets):
#         min_t = min(full_data[k].timestamps)
#         train_data[k].normalize_timestamps(min_t)
#         val_data[k].normalize_timestamps(min_t)
#         test_data[k].normalize_timestamps(min_t)

args.node_init_dim = node_features.shape[1]
args.node_embedding_dim = node_features.shape[1]
day_result = time.strftime("%Y%m%d", time.localtime(time.time()))
for rp in range(rp_times):

    print('repeat time',rp)
    args.seed = rp    ####
    seed = args.seed
    set_seed(seed)

    start_time=time.time()
    logger, time_now,day,based_path = set_logger(args.method,args.model, args.dataset, rp,args.filename_add, log_to_file)
   

    wandb.login()
    wandb.init(project = 'otg', config = vars(args),name = str(args.method) + '_'+ str(args.model) + '_' +str(args.dataset)+'_'+str(args.filename_add)+'_rp'+str(rp), reinit=True)
    warnings.filterwarnings('ignore')
    run_name = wandb.run.name

    Path("result/{}/{}".format(args.dataset,day_result)).mkdir(parents=True, exist_ok=True)
    logger.debug("./result/{}/{}/{}_{}.txt".format(args.dataset,day_result,args.method,args.filename_add))
    result_path = "./result/{}/{}/{}_{}.txt".format(args.dataset, day_result, args.method, args.filename_add)
    try:
        with open(result_path, "a+") as f:
            # Write content to the file
            f.write('\n')
            f.write(str(args))
            f.write("\n")
            f.write(time_now)
            f.write('\n')
            f.write("run_name: %s"%run_name)
            f.write("\n")
            f.write("seed: %d\n"%seed)
            f.flush()  # Ensure all data is written to the file
            print("Data written successfully")
    except Exception as e:
        print("An error occurred:", e)
    # f = open("./result/{}/{}/{}_{}.txt".format(args.dataset,day,args.method,args.filename_add),"a+")
    # f.write('\n')
    # f.write(str(args))
    # f.write("\n")
    # f.write(time_now)

    # f.write("\n")


    print(str(args))
  
    
    label_src = all_data.labels_src
    label_dst = all_data.labels_dst
    node_src = all_data.src
    node_dst = all_data.dst
    edge_timestamp = all_data.timestamps
    
    node_label = [-1 for i in range(all_data.n_unique_nodes + 1)]
    for i in range(len(label_src)):
        node_label[all_data.src[i]]=label_src[i]
        node_label[all_data.dst[i]]=label_dst[i]
    

    mean_time_shift_src, std_time_shift_src, mean_time_shift_dst, std_time_shift_dst = \
    computer_time_statics(train_data[0].src, train_data[0].dst, train_data[0].timestamps)

    args.time_shift = {'mean_time_shift_src': mean_time_shift_src, 'std_time_shift_src': std_time_shift_src, 
                       'mean_time_shift_dst': mean_time_shift_dst, 'std_time_shift_dst': std_time_shift_dst}
    logger.debug(str(args))
    # neighbor_finder = get_neighbor_finder(all_data, False) #### 
    neighbor_finder = get_neighbor_sampler(all_data, sample_neighbor_strategy=args.sample_neighbor_strategy,
                                                  time_scaling_factor=args.time_scaling_factor,seed=seed)
    
    sgnn = get_model(args, neighbor_finder, node_features, edge_features, label_src, label_dst)
    # prev_model = get_model(args, neighbor_finder, node_features, edge_features, label_src, label_dst)
    
    pre_model = None
    sgnn.to(args.device)
    wandb.watch(sgnn,wandb.config,log='all',log_freq=1)

  

    
    # LOSS = []
    val_n_acc_list,val_acc_list, val_ap_list, val_f1_list = [], [], [],[]
    if args.stop_method == 'loss':
        early_stopper = [EarlyStopMonitor(max_round=patience,higher_better=False) for i in range(args.num_datasets+1)]
    else:
        early_stopper = [EarlyStopMonitor(max_round=patience,higher_better=True) for i in range(args.num_datasets+1)]
    test_best=[0 for i in range(args.num_datasets)]
    test_neighbor_finder=[]

    if not os.path.exists(f'./checkpoints/{args.model}/'):
        os.makedirs(f'./checkpoints/{args.model}/')

    cur_train_data = None
    cur_test_data = None
    cur_val_data = None
    pre_train_data = None
    pre_data = None

    train_neighbor_finder_list = []
    test_neighbor_finder_list = []
    for task in range(0,args.num_datasets):
        wandb.log({"num_task":task})
        if args.memory_frac > 0:
            args.memory_size = int(args.memory_frac * len(train_data[task].src))
            print("the memory size is", args.memory_size)

        # initialize temporal graph
        if (task == 0) or (args.method != 'Joint'):
            cur_train_data = deepcopy(train_data[task])
            cur_test_data = deepcopy(test_data[task])
            cur_val_data = deepcopy(val_data[task])
            pre_data = deepcopy(full_data[task])
        else:
            cur_train_data.add_data(train_data[task])
            cur_test_data.add_data(test_data[task])
            cur_val_data.add_data(val_data[task])
            pre_data.add_data(full_data[task])
        
        if task > 0:
            pre_train_data = deepcopy(train_data[task-1])

        if len(cur_train_data.src) == 0 or len(cur_train_data.dst) == 0:
            print("Skipping training due to empty dataset.")
            continue 
        
        
        train_neighbor_finder = get_neighbor_sampler(cur_train_data, sample_neighbor_strategy=args.sample_neighbor_strategy,
                                                  time_scaling_factor=args.time_scaling_factor,seed=seed)
        test_neighbor_finder.append(get_neighbor_sampler(cur_train_data, sample_neighbor_strategy=args.sample_neighbor_strategy,
                                                  time_scaling_factor=args.time_scaling_factor,seed=seed))
        if args.method in ['ER']:
            if sgnn.memory_buffer is not None:
                print(task)
                cur_neighbor_data = cur_train_data.add_data(sgnn.memory_buffer)
                replay_neighbor_finder = get_neighbor_sampler(cur_neighbor_data, sample_neighbor_strategy=args.sample_neighbor_strategy,
                                                    time_scaling_factor=args.time_scaling_factor)
            else:
                replay_neighbor_finder = None
        
        full_neighbor_finder = get_neighbor_sampler(all_data, sample_neighbor_strategy=args.sample_neighbor_strategy,
                                                    time_scaling_factor=args.time_scaling_factor,seed=seed)
        if args.method == 'Joint':
            eval_neighbor_finder = get_neighbor_sampler(pre_data, sample_neighbor_strategy=args.sample_neighbor_strategy,
                                                    time_scaling_factor=args.time_scaling_factor,seed=seed)
        else:
            eval_neighbor_finder = get_neighbor_sampler(full_data[task], sample_neighbor_strategy=args.sample_neighbor_strategy,
                                                    time_scaling_factor=args.time_scaling_factor,seed=seed)
        
        train_neighbor_finder_list.append(train_neighbor_finder)   
        test_neighbor_finder_list.append(eval_neighbor_finder)

        sgnn.set_neighbor_finder(train_neighbor_finder)
        saved_model_paths = []
        mmd_time_epoch = 0
        epoch_end_time = 0
        if checkavaliable(args):
            sgnn = set_model(args,task)
            sgnn.eval()
        else:
            print("Please check avaliable model and dataset")
            sys.exit("Exiting program due to unavailable model or dataset.")
            for e in range(args.n_epoch):
                epoch_time = time.time()
                print("task:",task,"epoch:",e)
                logger.debug('task {} , start {} epoch'.format(task,e))
                num_batch = math.ceil(len(cur_train_data.src) / args.batch_size)
                loss_value = 0
                dis_loss = 0
                class_loss = 0
                mmd_loss = 0
                mmd_time = 0
                classifiy_loss = 0
                sgnn.train()
                for i in range(num_batch):
                    st_idx = i * args.batch_size
                    ed_idx = min((i + 1) * args.batch_size, len(cur_train_data.src))

                    src_batch = cur_train_data.src[st_idx:ed_idx]
                    dst_batch = cur_train_data.dst[st_idx:ed_idx]
                    edge_batch = cur_train_data.edge_idxs[st_idx:ed_idx]
                    timestamp_batch = cur_train_data.timestamps[st_idx:ed_idx]

                    if args.method  == 'NForGOT':
                        data_dict = sgnn.observe(src_batch, dst_batch, edge_batch, timestamp_batch, args.num_neighbors, pre_model,pre_train_data, st_idx,ed_idx,train_neighbor_finder_list,dataset_idx=task)
                        dis_loss += data_dict['dist']
                        class_loss += data_dict['class_dis']
                        mmd_loss += data_dict['mmd']
                        classifiy_loss += data_dict['loss']
                        mmd_time += data_dict['mmd_time']
                    else:
                        print('joint traning or formulate your own method here')
                        data_dict = sgnn.observe(src_batch, dst_batch, edge_batch, timestamp_batch, args.num_neighbors, dataset_idx=task)

                    loss_value += data_dict['total_loss']

                loss_value=loss_value / num_batch
                dis_loss = dis_loss /num_batch
                class_loss = class_loss / num_batch
                mmd_loss = mmd_loss / num_batch
                classifiy_loss = classifiy_loss / num_batch
                mmd_time_epoch += mmd_time
                
                print("Task: %d"%task," epoch: %d"%e,"  Total loss: %.4f"%loss_value)
                
                            
                # validation
                sgnn.eval()
                sgnn.set_neighbor_finder(eval_neighbor_finder)
                train_n_acc, train_acc, train_ap, train_f1, _ ,_ = eval_prediction(sgnn, train_data[task], task, task, args.batch_size, 'train', uml, eval_avg, args.multihead, args.num_class_per_dataset)

                val_n_acc, val_acc, val_ap, val_f1, val_loss,a = eval_prediction(sgnn, val_data[task], task, task, args.batch_size, 'val', uml, eval_avg, args.multihead, args.num_class_per_dataset)
                
                epoch_end_time += time.time()-epoch_time

                val_n_acc_list.append(val_n_acc)
                val_acc_list.append(val_acc)
                val_ap_list.append(val_ap)
                val_f1_list.append(val_f1)
            
                if args.average_stopper:
                    for k in range(task+1):
                        sgnn.set_neighbor_finder(test_neighbor_finder_list[k])
                        v_test_n_acc, v_test_acc, v_test_ap, v_test_f1,_,_= eval_prediction(sgnn, test_data[k], k, k, args.batch_size, 'test', uml, eval_avg, args.multihead, args.num_class_per_dataset)
            
                        val_test[k][task] = [v_test_n_acc, v_test_acc, v_test_ap, v_test_f1]
                    
                    sgnn.end_epoch()
                    
                    ## calculate for early stopper
                    v_ap, v_af ,v_BWT,_,_= calculate_avg_performance_forgetting(val_test,metrix,task+1)
                    
                    
                    print('test AP',v_ap['acc'])
                    if args.stop_method == 'acc':
                        stop = v_ap['acc'][task]
                    elif args.stop_method == 'n_acc':
                        stop = v_ap['n_acc'][task]
                    elif args.stop_method == 'ap':
                        stop = v_ap['ap'][task]
                    
                    # if task ==0: 
                    #     wandb.log({f'Total_loss_{task}':loss_value,f'dis_loss_{task}':dis_loss,f'class_loss_{task}':class_loss,
                    #     f'mmd_loss_{task}':mmd_loss,f'classifiy_loss_{task}':classifiy_loss,f'epoch_step_{task}':e,f'mmd_time_{task}':mmd_time, 
                    #     f"val_n_acc_{task}": val_n_acc,f'val_acc_{task}':val_acc,f'val_ap_{task}':val_ap,f'val_f1_{task}':val_f1,
                    #     f'val_loss_{task}':val_loss,f"v_test_ap_acc_{task}": v_ap['acc'][0],f"v_test_ap_n_acc_{task}":v_ap['n_acc'][0]})
                    # else:
                    #     wandb.log({f'Total_loss_{task}':loss_value,f'dis_loss_{task}':dis_loss,f'class_loss_{task}':class_loss,
                    #     f'mmd_loss_{task}':mmd_loss,f'classifiy_loss_{task}':classifiy_loss,f'epoch_step_{task}':e,f'mmd_time_{task}':mmd_time, 
                    #     f"val_n_acc_{task}": val_n_acc,f'val_acc_{task}':val_acc,f'val_ap_{task}':val_ap,f'val_f1_{task}':val_f1,
                    #     f'val_loss_{task}':val_loss,f"v_test_ap_acc_{task}": v_ap['acc'][-1],f"v_test_ap_n_acc_{task}":v_ap['n_acc'][-1],f"v_test_af_acc_{task}":v_af['acc'][-1],
                    #     f"v_test_af_n_acc_{task}":v_af['n_acc'][-1],f"v_test_bwt_acc_{task}":v_BWT['acc'][-1],f"v_test_bwt_n_acc_{task}":v_BWT['n_acc'][-1]})
                
                else:
                    v_test_n_acc, v_test_acc, v_test_ap, v_test_f1,_,b = eval_prediction(sgnn, test_data[task], task, task, args.batch_size, 'test', uml, eval_avg, args.multihead, args.num_class_per_dataset)
                    
                    sgnn.end_epoch()
                    
                    # early stopper
                    if args.stop_method == 'loss':
                        stop = val_loss.cpu().numpy()
                    elif args.stop_method == 'acc':
                        stop = v_test_acc
                    elif args.stop_method == 'n_acc':
                        stop = v_test_n_acc
                    elif args.stop_method == 'ap':
                        stop = v_test_ap
                    

                ######### set this as the early stopper ########### can influence the best model
                if early_stopper[task].early_stop_check(stop): 
                    logger.info('No improvment over {} epochs, stop training'.format(early_stopper[task].max_round))
                    logger.info(f'Loading the best model at epoch {early_stopper[task].best_epoch}')
                    best_model_path = get_checkpoint_path(based_path,early_stopper[task].best_epoch,task, uml)
                    print("best_model_path:", best_model_path)
                    sgnn = torch.load(best_model_path)
                    logger.debug(f'Loaded the best model at epoch {early_stopper[task].best_epoch} for inference')
                    sgnn.eval()
                    with open(result_path, "a+") as f:
                        f.write("task %d:  v_test_n_acc:%.4f   v_test_acc:%.4f   v_test_ap:%.4f   v_test_f1:%.4f  \n "%(task,v_test_n_acc,v_test_acc,v_test_ap,v_test_f1))
                        f.flush()
                        
                    # Delete previously saved models except the best one
                    for model_path in saved_model_paths:
                        if model_path != best_model_path:
                            try:
                                os.remove(model_path)
                                logger.debug(f'Removed model at {model_path}')
                            except OSError as e:
                                logger.error(f'Error removing {model_path}: {e}')
                    
                    break 
                else:
                    if e == args.n_epoch-1:
                        logger.info('not enough epochs, stop training')
                        with open(result_path, "a+") as f:
                            f.write('not enough epochs, stop training \n')
                            f.flush()
                    model_path = get_checkpoint_path(based_path, e, task, uml)
                    torch.save(sgnn, model_path,pickle_module=dill)
                    saved_model_paths.append(model_path)
                    if e == 200:
                        for i in saved_model_paths[:150]:
                            try:
                                os.remove(i)
                                saved_model_paths.remove(i)
                            except OSError as e:
                                logger.error(f'Error removing {i}: {e}')
            
                        
                    
            # prepare for the next task, store the model/prototype/reused data
            if args.method == 'NForGOT':
                torch.cuda.empty_cache()
                sgnn.get_prototype(train_data[task], task)
                with torch.no_grad():
                    pre_model = copy.deepcopy(sgnn.net).to(args.device)
            else:
                print("joint training or please set up for your own method")

        ### Test 
        if args.online:
            for k in range(task+1):
                if not args.multihead:
                    test_n_acc, test_acc, test_ap, test_f1,_,_ = eval_prediction(sgnn, test_data[k], task, k, args.batch_size, 'test', uml, eval_avg, args.multihead, args.num_class_per_dataset)
                else:
                    test_n_acc, test_acc, test_ap, test_f1, = eval_prediction(sgnn, test_data[k], k, k, args.batch_size, 'test', uml, eval_avg, args.multihead, args.num_class_per_dataset)
        
                test_best[k]=max(test_best[k], test_n_acc)
                task_acc_vary[k][task]+=test_n_acc
                task_acc_vary_cur[k][task]=test_n_acc
                print("online task %d acc: "%(k)+str(task_acc_vary_cur[k][task]))      
                
        else:
            for k in range(task+1):
                
                sgnn.set_neighbor_finder(test_neighbor_finder_list[k])

                if not args.multihead:
                    test_n_acc, test_acc, test_ap, test_f1,_,_= eval_prediction(sgnn, test_data[k], task, k, args.batch_size, 'test', uml, eval_avg, args.multihead, args.num_class_per_dataset)
                else:
                    test_n_acc, test_acc, test_ap, test_f1,_ ,_= eval_prediction(sgnn, test_data[k], k, k, args.batch_size, 'test', uml, eval_avg, args.multihead, args.num_class_per_dataset)
            
                test_best[k] = max(test_best[k], test_acc)
                task_acc_vary[k][task] += [test_n_acc, test_acc, test_ap, test_f1]
                task_acc_vary_cur[k][task] = [test_n_acc, test_acc, test_ap, test_f1]
                  
                print("task %d acc: "%(k)+str(task_acc_vary_cur[k][task]))
        
        print('task ',task,':')
        print('test_n_acc:',test_n_acc,'test_acc:',test_acc,'test_ap:',test_ap,'test_f1:',test_f1)
    
    all_time=time.time()-start_time
    print("all_time: ", all_time/3600)
    
 

    
    # Summarize the results
    
    print('best performance: ',test_best)

    avg_performance, all_forgetting ,all_BWT,all_forgetting_max,only_forgetting= calculate_avg_performance_forgetting(task_acc_vary_cur,metrix,len(task_acc_vary_cur))
    avg_performance_all.append(avg_performance)
    avg_forgetting_all.append(all_forgetting)
    forgetting_max_all.append(all_forgetting_max)
    bwt_all.append(all_BWT)
    only_forgetting_all.append(only_forgetting)    
    
    ############## write the final result to the file ##############
    try:
        with open(result_path, "a+") as f:
            f.write("all_time: "+str(all_time/3600))
            f.write("\n")
            f.write('rp time: '+str(rp)+'\n')
            f.write('avg_performance: \n')
            for key, values in avg_performance.items():
                f.write(f"{key}: {values}\n")
                
            
            f.write('avg_forgetting: \n')
            for key, values in all_forgetting.items():
                f.write(f"{key}: {values}\n")
               
            f.write('BWT: \n')
            for key, values in all_BWT.items():
                f.write(f"{key}: {values}\n")
                

            f.write('forgetting_max: \n')
            for key, values in all_forgetting_max.items():
                f.write(f"{key}: {values}\n")
                
            f.write('only_forgetting: \n')
            for key, values in only_forgetting.items():
                f.write(f"{key}: {values}\n")
                
            f.write("task_acc_vary_cur \n")
            for i in range(args.num_datasets):  
                f.write('task %d: '%(i)+str(task_acc_vary_cur[i][i:]))
                f.write("\n")
            f.flush()
    except Exception as e:
        print('write final result error: ',e)
    

   
 

overall_time +=all_time 
print("overall_time: ", overall_time/3600/rp_times)
overall_ap = average_result(avg_performance_all)
averall_af = average_result(avg_forgetting_all)
overall_af_max = average_result(forgetting_max_all)
overall_bwt = average_result(bwt_all)
overall_onlyforgetting = average_result(only_forgetting_all)

try:
    with open(result_path, "a+") as f:
        f.write("overall_time: "+str(overall_time/3600/rp_times))
        f.write(str(args))
        f.write("\n")
        f.write('finish %d times repeat times'%rp_times)
        f.write(time_now)

        f.write('Overall avg_performance: \n')
        for key, values in overall_ap.items():
            f.write(f"{key}: {values}\n")
            
        f.write('Overall avg_forgetting: \n')
        for key, values in averall_af.items():
            f.write(f"{key}: {values}\n")
           
        f.write('Overall avg_forgetting_max: \n')
        for key, values in overall_af_max.items():
            f.write(f"{key}: {values}\n")
            

        f.write('Overall BWT: \n')
        for key, values in overall_bwt.items():
            f.write(f"{key}: {values}\n")
            

        f.write('Overall only_forgetting: \n')
        for key, values in overall_onlyforgetting.items():
            f.write(f"{key}: {values}\n")
            


        f.write("task_acc_vary \n")

        for i in range(args.num_datasets):
            for j in range(i,args.num_datasets):  
                if rp_times != 0:
                    task_acc_vary[i][j] = [x / rp_times for x in task_acc_vary[i][j]]  
            f.write("task %d: "%(i)+str(task_acc_vary[i][i:]))
            f.write("\n")


        f.write("task_acc_vary_cur \n")
        for i in range(args.num_datasets):
            f.write('task %d: '%(i)+str(task_acc_vary_cur[i][i:]))
            f.write("\n")
            

        f.write("\n ========================= \n")
        f.flush()
except Exception as e:
    print('write overall result error: ',e)

wandb.finish()

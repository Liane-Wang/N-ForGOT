import torch
import copy
import os
import time
import math
from torch.autograd import Variable
import torch.nn.functional as F
from dgl.nn.pytorch import edge_softmax, GATConv
import torch.nn as nn
import numpy as np
from models.Backbone import TemporalGNNClassifier
from methods.NForGOT_utils import Data, Memory
from copy import deepcopy
import wandb 


class General_MMD_loss(torch.nn.Module):
    def __init__(self, kernel_mul=2.0, kernel_num=5):
        super(General_MMD_loss, self).__init__()
        self.kernel_num = kernel_num
        self.kernel_mul = kernel_mul
        self.fix_sigma = None

    def guassian_kernel(self, source, target, kernel_mul=2.0, kernel_num=5, fix_sigma=None):
        n_samples = int(source.size()[0]) + int(target.size()[0])
        print(source.shape,target.shape)
        total = torch.cat([source, target], dim=0)
        eps = 1e-6
        total = total.squeeze(0).reshape(-1, 300)
        total0 = total.unsqueeze(0).expand(int(total.size(0)), int(total.size(0)), int(total.size(1)))
        total1 = total.unsqueeze(1).expand(int(total.size(0)), int(total.size(0)), int(total.size(1)))
        L2_distance = ((total0 - total1) ** 2).sum(2)
        if fix_sigma:
            bandwidth = fix_sigma
        else:
            bandwidth = torch.sum(L2_distance.data) / (n_samples ** 2 - n_samples)
        bandwidth /= kernel_mul ** (kernel_num // 2)
        bandwidth_list = [bandwidth * (kernel_mul ** i) for i in range(kernel_num)]
        bandwidth_list = [b + eps for b in bandwidth_list]
        kernel_val = [torch.exp(-L2_distance / bandwidth_temp) for bandwidth_temp in bandwidth_list]
        return sum(kernel_val)

    def forward(self, source, target):
        batch_size = int(source.size()[0])
        kernels = self.guassian_kernel(source, target, kernel_mul=self.kernel_mul, kernel_num=self.kernel_num, fix_sigma=self.fix_sigma)
        XX = kernels[:batch_size, :batch_size]
        YY = kernels[batch_size:, batch_size:]
        XY = kernels[:batch_size, batch_size:]
        YX = kernels[batch_size:, :batch_size]
        loss = torch.mean(XX + YY - XY - YX)
        return loss

class LinearTimeMMDLoss(torch.nn.Module):
    def __init__(self, kernel_mul=2.0, kernel_num=5):
        super(LinearTimeMMDLoss, self).__init__()
        self.kernel_mul = kernel_mul
        self.kernel_num = kernel_num

    def gaussian_kernel(self, x1, x2, y1, y2, kernel_mul=2.0, kernel_num=5):
        eps = 1e-6
        # Calculate the Gaussian kernel values for the pairs (x1, x2) and (y1, y2)
        L2_distance_xx = torch.sum((x1 - x2) ** 2)
        L2_distance_yy = torch.sum((y1 - y2) ** 2)
        L2_distance_xy = torch.sum((x1 - y2) ** 2)
        L2_distance_yx = torch.sum((x2 - y1) ** 2)

        bandwidth = (L2_distance_xx + L2_distance_yy + L2_distance_xy + L2_distance_yx) / 4
        bandwidth /= kernel_mul ** (kernel_num // 2)
        bandwidth_list = [bandwidth * (kernel_mul ** i) for i in range(kernel_num)]
        bandwidth_list = [b + eps for b in bandwidth_list]

        kernel_val_xx = [torch.exp(-L2_distance_xx / b) for b in bandwidth_list]
        kernel_val_yy = [torch.exp(-L2_distance_yy / b) for b in bandwidth_list]
        kernel_val_xy = [torch.exp(-L2_distance_xy / b) for b in bandwidth_list]
        kernel_val_yx = [torch.exp(-L2_distance_yx / b) for b in bandwidth_list]
        return sum(kernel_val_xx) + sum(kernel_val_yy) - sum(kernel_val_xy) - sum(kernel_val_yx)

    def forward(self, source, target):
        m = source.size(0)
        m2 = m // 2

        # Create indices for the odd and even pairs
        indices_odd = torch.arange(0, m, 2, device=source.device)
        indices_even = torch.arange(1, m, 2, device=source.device)

        x_odd = source[indices_odd]
        y_odd = target[indices_odd]
        x_even = source[indices_even]
        y_even = target[indices_even]

        # Ensure we only use m2 pairs
        if x_even.size(0) < m2:
            m2 = x_even.size(0)

        x_odd = x_odd[:m2]
        y_odd = y_odd[:m2]
        x_even = x_even[:m2]
        y_even = y_even[:m2]

        # print('y_even',y_even)  
        # Compute h for each pair
        h_values = torch.stack([
            self.gaussian_kernel(x_odd[i], x_even[i], y_odd[i], y_even[i], self.kernel_mul, self.kernel_num)
            for i in range(m2)
        ])
        # print('h_values',h_values)

        return h_values.mean()
        
class NForGOT(torch.nn.Module):
    """
    neighbor_finder: NeighborFinder
    node_features: torch.Tensor
    edge_features: torch.Tensor
    src_label: List[int]
    dst_label: List[int]
    """
    def __init__(self, args, neighbor_finder, node_features, edge_features, src_label, dst_label):
        super(NForGOT, self).__init__()
        self.args = args
        print('Initialization NForGOT method')
        if args.supervision == 'supervised':
            self.net = TemporalGNNClassifier(args, neighbor_finder, node_features, edge_features, src_label, dst_label)
        
        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=args.lr,weight_decay=
                                          0.0)
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=100, gamma=0.5) #****
       
        self.args = args
        self.activation = F.elu

        # self.net.apply(kaiming_normal_init)                
        self.memory = Memory()
        self.src_label = torch.tensor(src_label).to(args.device)
        self.dst_label = torch.tensor(dst_label).to(args.device)
        self.current_class_pro = {}
        self.class_prototype = {}


    def euclidean_dist(self,x, y):
        # x: N x D query
        # y: M x D prototype
        n = x.size(0)
        m = y.size(0)
        d = x.size(1)
        assert d == y.size(1)

        x = x.unsqueeze(1).expand(n, m, d)
        y = y.unsqueeze(0).expand(n, m, d)

        # return 1 / (1e-6 + torch.pow(x - y, 2).sum(2)) ## relevance score
        return torch.pow(x - y, 2).sum(2)  # N x M
  

    def distance_matrix_loss(self,dist_matrix1, dist_matrix2):
        """
        Calculate the Mean Squared Error (MSE) between two distance matrices.
        Parameters:
            dist_matrix1 (torch.Tensor): The first distance matrix, size [n, m]
            dist_matrix2 (torch.Tensor): The second distance matrix, size [n, m]
        Returns:
            torch.Tensor: The MSE loss between the two distance matrices.
        """
        loss = F.mse_loss(dist_matrix1, dist_matrix2)
        return loss

    def MultiClassCrossEntropy(self,logits, labels, T): ## old (log logits)
        labels = Variable(labels.data, requires_grad=False).cuda(self.args.device)
        outputs = torch.log_softmax(logits/T, dim=1)   # compute the log of softmax values
        labels = torch.softmax(labels/T, dim=1)
        outputs = torch.sum(-outputs * labels, dim=1, keepdim=False)
        outputs = torch.mean(outputs, dim=0)
        return outputs
    


    def MultiClassCrossEntropy_class(self,current, record, T=2.0):
        p = F.softmax(current / T, dim=1).cuda(self.args.device)
        q = F.softmax(record / T, dim=1).cuda(self.args.device)
        # soft_loss = -torch.mean(torch.sum(q * p, dim=1))
        soft_loss = 1/(1e-6 + torch.mean(p * torch.log(p/q)))
        return soft_loss
    
    def compute_relevance_scores(self,old_protos, new_protos):
        
        relevance_scores = torch.mm(old_protos, new_protos.t())  
        relevance_scores = F.softmax(relevance_scores, dim=1)  
        return relevance_scores
    
    
    def class_loss(self,emb_src,pre_emb_src,edges,task):
        
        pre_current_class_pro = self.get_current_prototype(pre_emb_src,edges,task)
        current_class_pro = self.get_current_prototype(emb_src,edges,task)

        ### need clear boundary 
        end = (task + 1) * self.args.num_class_per_dataset
        start_pre = (task-1) * self.args.num_class_per_dataset 
        end_pre = task *  self.args.num_class_per_dataset 

        ## last task
        pre_pro = torch.stack(list(self.class_prototype[str(keys)] for keys in range(start_pre, end_pre))).squeeze(1).to(self.args.device)

        ## only current task
        valid_keys = [str(key) for key in range(end_pre, end) if str(key) in current_class_pro and current_class_pro[str(key)].numel() > 0]

        
        selected_current_class_pro = [current_class_pro[key] for key in valid_keys]
        if selected_current_class_pro:
            selected_current_class_pro= torch.cat(selected_current_class_pro, dim=0) ##  size: class number * emb size
        else:
            selected_current_class_pro = torch.empty(0)  #

        selected_pre_class_pro = [pre_current_class_pro[key] for key in valid_keys]
        if selected_pre_class_pro:
            selected_pre_class_pro = torch.cat(selected_pre_class_pro, dim=0)
        else:
            selected_pre_class_pro = torch.empty(0)

       
        dist_matrix_current =  self.euclidean_dist(selected_current_class_pro,pre_pro)  # ** pre all task

        dist_matrix_pre  = self.euclidean_dist(selected_pre_class_pro,pre_pro)
        
        # loss = F.mse_loss(dist_matrix_current, dist_matrix_pre)
        diff_matrix = dist_matrix_current - dist_matrix_pre
        # loss = torch.norm(diff_matrix, p=2)
         
        # reg_loss = lambda_reg * (torch.norm(current_class_pro_tensors, p=2) + torch.norm(current_class_pre_nn, p=2))
        # loss = 0.01* torch.norm(diff_matrix, p='fro') + reg_loss
        loss = torch.norm(diff_matrix, p='fro')
       

        del pre_current_class_pro,current_class_pro,selected_pre_class_pro,selected_current_class_pro

        return loss

    
    def exponential_decay(self,initial_value, task_index,decay_rate=0.8):
        return 1- initial_value * (decay_rate ** task_index)
    
                
    def observe(self, src_nodes, dst_nodes, edges, edge_times, n_neighbors, prev_model, predata, st_idx,ed_idx,train_neighbor_finder_list,dataset_idx=None):
       
        self.current_task = dataset_idx
        data_dict = {}
        loss,logits = self.net(src_nodes, dst_nodes, edges, edge_times, n_neighbors, dataset_idx)
        data_dict['loss'] = loss.item()
       

        if dataset_idx > 0:

            prev_model = prev_model.to(self.args.device)
            
            with torch.no_grad():
                _,target = prev_model(src_nodes, dst_nodes, edges, edge_times, n_neighbors, dataset_idx,pre=True)
                pre_emb_src,pre_emb_dst= prev_model.get_embeddings(src_nodes, dst_nodes, edges, edge_times, n_neighbors)
                emb_src,emb_dst,emb_src_neighbor = self.net.get_embeddings_with_neighbors(src_nodes, dst_nodes, edges, edge_times, n_neighbors) 
              
            if isinstance(target, tuple):
                target = target[0]  ## only src logits
            if isinstance(logits, tuple):
                logits = logits[0]  ## only src logits

         
            if target.size(1) > self.args.num_class_per_dataset:
                target = target[:,-self.args.num_class_per_dataset:]

            logits = logits[:,-self.args.num_class_per_dataset:]
     
            ## new classes in task t-1 vs new classes in tast t
            dis_loss = self.MultiClassCrossEntropy(logits, target, self.args.T)   
           
            ## class loss
            class_loss = self.class_loss(emb_src,pre_emb_src,edges,dataset_idx)
            
            ## MMD loss 
            pre_batch_data = self.memory.pre_batch(predata,st_idx,ed_idx)
            with torch.no_grad():
                self.set_neighbor_finder(train_neighbor_finder_list[dataset_idx-1])
                memory_src, _,memory_src_neighbor = self.net.get_embeddings_with_neighbors(pre_batch_data.src,
                                                                                                    pre_batch_data.dst,
                                                                                                    pre_batch_data.edge_idxs,
                                                                                                    pre_batch_data.timestamps, n_neighbors) 
                self.set_neighbor_finder(train_neighbor_finder_list[dataset_idx])
                                                                                                   
            if self.args.linear_mmd:
                MMD_loss = LinearTimeMMDLoss()
            else:
                MMD_loss = General_MMD_loss()
            
            if self.args.model == 'DyGFormer':
                n_neighbors = min(emb_src_neighbor.size(1),memory_src_neighbor.size(1))

            src_tree = torch.cat((emb_src_neighbor[:, :n_neighbors, :], emb_src.unsqueeze(1)), dim=1)
            memory_src_tree = torch.cat((memory_src_neighbor[:, :n_neighbors, :], memory_src.unsqueeze(1)), dim=1)
           
            mmd_time = time.time()
            mmd_loss = MMD_loss(src_tree, memory_src_tree) ## tree structure

            mmd_end = time.time()

            del memory_src,memory_src_neighbor,src_tree,memory_src_tree

            loss = loss + self.args.alpha*(self.args.distill_loss*dis_loss + \
                self.args.class_loss * class_loss) + self.args.beta*(self.args.mmd_loss * mmd_loss)
       

        self.optimizer.zero_grad()
        loss.backward()
        
        data_dict['dist'] = dis_loss.item() if dataset_idx > 0 else 0
        data_dict['class_dis'] = class_loss.item() if dataset_idx > 0 else 0
        data_dict['mmd'] = mmd_loss.item() if dataset_idx > 0 else 0
        data_dict['total_loss'] = loss.item()
        data_dict['mmd_time'] = mmd_end-mmd_time if dataset_idx > 0 else 0

        self.optimizer.step()
        
        return data_dict

    
    def get_current_prototype(self,emb_src,edges,task,current=True):
        
        train_class_src = self.get_class(self.src_label[edges])
         

        prototype_src = {}
        for key, indices in train_class_src.items():
            if indices.numel() > 0:  # Ensure there are elements to avoid empty selection
                prototype_src[str(key)] = torch.mean(emb_src[indices], dim=0, keepdim=True).to(self.args.device)

        
        return prototype_src
    
    def get_class(self, y_label):
      
        if isinstance(y_label, np.ndarray):
            y_label = torch.from_numpy(y_label).to(self.args.device) 

        labels, inverse = torch.unique(y_label, sorted=True, return_inverse=True)
        class_by_id = {str(label.item()): (inverse == idx).nonzero(as_tuple=True)[0] for idx, label in enumerate(labels)}

        return class_by_id

    
    def get_prototype(self,data,task):
        batch_size = self.args.batch_size
        
        end = (task + 1) * self.args.num_class_per_dataset
        start = task *  self.args.num_class_per_dataset 
        prototype_src = {str(key): [] for key in range(start, end)}

        with torch.no_grad():
            num_batch = math.ceil(len(data.src) / batch_size)
            for i in range(num_batch):
                st_idx = i * batch_size
                ed_idx = min((i + 1) * batch_size, len(data.src))
                emb_src,_ = self.net.get_embeddings(data.src[st_idx:ed_idx], data.dst[st_idx:ed_idx], data.edge_idxs[st_idx:ed_idx], data.timestamps[st_idx:ed_idx], self.args.num_neighbors)
                train_class_src = self.get_class(data.labels_src[st_idx:ed_idx])
               
                for key in train_class_src.keys():
                    indices = torch.tensor(train_class_src[str(key)])
                    mask = (indices >= start) & (indices < end)
                    valid_indices = indices[mask] 
                    if len(valid_indices) > 0:
                        prototype_src[str(key)].append(torch.mean(emb_src[valid_indices], dim=0))
                    
        
        for key, embed_list in prototype_src.items():
            if embed_list:
                self.class_prototype[str(key)] = torch.mean(torch.stack(embed_list), dim=0, keepdim=True).to(self.args.device)
        
        del prototype_src,emb_src
        return self.class_prototype


    def detach_memory(self):
        self.net.detach_memory()

    def end_epoch(self):
        self.scheduler.step()

    def get_acc(self, x, y):
        output = self.net(x)
        _, pred = torch.max(output, 1)
        correct = (pred == y).sum().item()
        return correct
    
    def adaptation(self,task):
        self.net.adaptation(task)
        return
    
    def get_logits(self, src_nodes, dst_nodes, edges, edge_times, n_neighbors, dataset_idx):
        src_logits, dst_logits,loss = self.net(src_nodes, dst_nodes, edges, edge_times, n_neighbors, dataset_idx, return_logits=True)
        return src_logits, dst_logits,loss
    
    def end_dataset(self, train_data, args):
        return
    

    def back_up_memory(self):
        return

    def restore_memory(self, back_up):
        return
   
    def update_memory(self,train_data,st_idx,ed_idx,mode):
        self.memory.pre_batch(train_data,st_idx,ed_idx)
        
    def get_replay_data(self,train_data,size,mode):
        if mode == 'random':
            sample_idx = np.random.choice(range(len(train_data.src)),size,replace = False)

        self.memory.update_memory(sample_idx,train_data)
        self.memory.get_data(size,mode)
    
    def update_cur_train_data(self,data):
        data = self.memory.update_data(data)
        return data

    def get_parameters(self):
        return self.net.parameters()

    def get_optimizer(self):
        return self.optimizer

    def get_scheduler(self):
        return self.scheduler

    def get_model(self):
        return self.net
    
    def set_neighbor_finder(self, neighbor_finder):
        self.net.set_neighbor_finder(neighbor_finder)

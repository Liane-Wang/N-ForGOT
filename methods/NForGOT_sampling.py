import torch
import copy
import os
import time
from torch.autograd import Variable
import torch.nn.functional as F
from dgl.nn.pytorch import edge_softmax, GATConv
import torch.nn as nn
import numpy as np
from models.Backbone import TemporalGNNClassifier
from methods.Ours_utils import Data, Memory
from copy import deepcopy
import wandb 

# def MultiClassCrossEntropy_class(logits, labels, T):
    
#     outputs = torch.log_softmax(logits/T, dim=1)   # compute the log of softmax values
#     labels = torch.softmax(labels/T, dim=1) 
#     outputs = torch.sum(outputs * labels, dim=1, keepdim=False)
#     outputs = -torch.mean(outputs, dim=0, keepdim=False)
#     return outputs





class MMD_loss(torch.nn.Module):
    def __init__(self, kernel_mul=2.0, kernel_num=5):
        super(MMD_loss, self).__init__()
        self.kernel_num = kernel_num
        self.kernel_mul = kernel_mul
        self.fix_sigma = None

    def guassian_kernel(self, source, target, kernel_mul=2.0, kernel_num=5, fix_sigma=None):
        n_samples = int(source.size()[0]) + int(target.size()[0])
        total = torch.cat([source, target], dim=0)
        eps = 1e-6
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
        # print('kernel_val_yx',kernel_val_yx)
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
        
class Ours_sampling(torch.nn.Module):
    """
        LwF baseline for NCGL tasks

        :param model: The backbone GNNs, e.g. GCN, GAT, GIN, etc.
        :param task_manager: Mainly serves to store the indices of the output dimensions corresponding to each task
        :param args: The arguments containing the configurations of the experiments including the training parameters like the learning rate, the setting confugurations like class-IL and task-IL, etc. These arguments are initialized in the train.py file and can be specified by the users upon running the code.

        """
    def __init__(self, args, neighbor_finder, node_features, edge_features, src_label, dst_label):
        super(Ours_sampling, self).__init__()
        self.args = args
        print('initialization')
        if args.supervision == 'supervised':
            self.net = TemporalGNNClassifier(args, neighbor_finder, node_features, edge_features, src_label, dst_label)
        
        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=args.lr,weight_decay=
                                          0.0)
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=50, gamma=0.5) #****
       
        self.args = args
        self.activation = F.elu

        # self.net.apply(kaiming_normal_init)                
        self.memory = Memory()
        self.src_label = torch.tensor(src_label).to(args.device)
        self.dst_label = torch.tensor(dst_label).to(args.device)
        self.current_class_pro = {}
        self.class_prototype = {}

        # setup losses
        # self.ce = torch.nn.functional.cross_entropy

        # self.current_task = 0
        # self.n_classes = 5
        # self.n_known = 0
        # self.prev_model = None

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

    def MultiClassCrossEntropy(self,logits, labels, T):
        labels = Variable(labels.data, requires_grad=False).cuda(self.args.device)
        outputs = torch.log_softmax(logits/T, dim=1)   # compute the log of softmax values
        labels = torch.softmax(labels/T, dim=1)
        outputs = torch.sum(-outputs * labels, dim=1, keepdim=False)
        outputs = torch.mean(outputs, dim=0)
        return outputs

    def MultiClassCrossEntropy_class(self,current, record, T=2.0):
        # p = F.log_softmax(student_output / T, dim=1)
        p = F.softmax(current / T, dim=1).cuda(self.args.device)
        q = F.softmax(record / T, dim=1).cuda(self.args.device)
        # soft_loss = -torch.mean(torch.sum(q * p, dim=1))
        soft_loss = 1/(1e-6 + torch.mean(p * torch.log(p/q)))
        return soft_loss
    
    def compute_relevance_scores(self,old_protos, new_protos):
        # 使用余弦相似度作为相关性的度量
        relevance_scores = torch.mm(old_protos, new_protos.t())  # 转置新类原型以进行矩阵乘法
        relevance_scores = F.softmax(relevance_scores, dim=1)  # 归一化分数以便用作权重
        return relevance_scores
    
    # def class_dist(self,emb_src,emb_dst,task):
        
    #     emb = torch.cat((emb_src, emb_dst), 0)
    #     current_class_pro_tensors = torch.stack(list(self.current_class_pro.values())).squeeze(1).to(emb.device)
        
    #     current_class_relavance = self.euclidean_dist(emb, current_class_pro_tensors[-3:])

    #     pre_pro = torch.stack(list(self.class_prototype.values())).to(self.args.device)
    #     pre_pro = pre_pro.squeeze(1)
    #     pre_class_relavance = self.euclidean_dist(emb, pre_pro) ## batch size * class number
    
    #     dis_loss = 0 
    #     for i in range(self.args.class_loss['num_task']):
    #         if i < task:
    #             pre_range = self.args.num_class_per_dataset * i
    #             class_range = self.args.num_class_per_dataset *(i+1)
               
    #             dis_loss +=self.MultiClassCrossEntropy_class(current_class_relavance, pre_class_relavance[:,pre_range:class_range], self.args.distill_loss['T_class'])
    #         # print('dis_loss',dis_loss)
    #     return dis_loss/int(self.args.class_loss['num_task'])
    
    def class_loss(self,emb_src,emb_dst,task):
        
        emb = torch.cat((emb_src, emb_dst), 0)
        # current_class_pro_tensors = torch.stack(list(self.current_class_pro.values())).squeeze(1).to(emb.device)
        # current_class_pre_nn = torch.stack(list(self.pre_current_class_pro.values())).squeeze(1).to(emb.device)
        #
        ### need clear boundary 
        # start = task * self.args.num_class_per_dataset
        end = (task + 1) * self.args.num_class_per_dataset
        start_pre = (task-1) * self.args.num_class_per_dataset 
        end_pre = task *  self.args.num_class_per_dataset 

        # valid_keys = [str(key) for key in range(start, end) if str(key) in self.current_class_pro and self.current_class_pro[str(key)].numel() > 0]
        # print(list(self.current_class_pro.keys()))
        # if valid_keys is None:
        #     return
        # selected_current_class_pro = [self.current_class_pro[key] for key in valid_keys]
    
        # if selected_current_class_pro:
        #     current_class_pro_tensors = torch.cat(selected_current_class_pro, dim=0)
        # else:
        #     current_class_pro_tensors = torch.empty(0)  #

        # current_class_pre = [self.pre_current_class_pro[key] for key in valid_keys]
        # if current_class_pre:
        #     current_class_pre_nn = torch.cat(current_class_pre, dim=0)
        # else:
        #     current_class_pre_nn = torch.empty(0)
       
        # selected_class_prototype = [self.class_prototype[key] for key in valid_keys if key in self.class_prototype]
        # if selected_class_prototype:
        #     pre_pro = torch.cat(selected_class_prototype, dim=0)
        # else:
        #     pre_pro = torch.empty(0)  

        pre_pro = torch.stack(list(self.class_prototype[str(keys)] for keys in range(start_pre, end_pre))).squeeze(1).to(self.args.device)
        # print('pre_pro',pre_pro.size())
        # print('current_class_pre_nn',current_class_pre_nn.size())
        ### 
        valid_keys = [str(key) for key in range(start_pre, end) if str(key) in self.current_class_pro and self.current_class_pro[str(key)].numel() > 0]

        
        selected_current_class_pro = [self.current_class_pro[key] for key in valid_keys]
        if selected_current_class_pro:
            current_class_pro_tensors = torch.cat(selected_current_class_pro, dim=0)
        else:
            current_class_pro_tensors = torch.empty(0)  #

        current_class_pre = [self.pre_current_class_pro[key] for key in valid_keys]
        if current_class_pre:
            current_class_pre_nn = torch.cat(current_class_pre, dim=0)
        else:
            current_class_pre_nn = torch.empty(0)

        # pre_pro = torch.stack(list(self.class_prototype[key] for key in valid_keys if key in self.class_prototype)).squeeze(1).to(self.args.device)
        print('pre_pro',pre_pro.size())
        print('current_class_pre_nn',current_class_pre_nn.size())
        ###
        dist_matrix_current =  self.euclidean_dist(current_class_pro_tensors,pre_pro)  # ** pre all task

        dist_matrix_pre  = self.euclidean_dist(current_class_pre_nn,pre_pro)
        # loss = F.mse_loss(dist_matrix_current, dist_matrix_pre)
        diff_matrix = dist_matrix_current - dist_matrix_pre
        # loss = torch.norm(diff_matrix, p=2)
        # print('class loss',loss)
        lambda_reg = 0.01
        # reg_loss = lambda_reg * (torch.norm(current_class_pro_tensors, p=2) + torch.norm(current_class_pre_nn, p=2))
        # loss = 0.01* torch.norm(diff_matrix, p='fro') + reg_loss
        loss = 0.01 * torch.norm(diff_matrix, p='fro')
        return loss

    
    def exponential_decay(self,initial_value, task_index,decay_rate=0.8):
        return 1- initial_value * (decay_rate ** task_index)
    
                
    def observe(self, src_nodes, dst_nodes, edges, edge_times, n_neighbors, prev_model,dataset_idx=None):
        self.current_task = dataset_idx
        data_dict = {}
        loss,logits = self.net(src_nodes, dst_nodes, edges, edge_times, n_neighbors, dataset_idx)
        data_dict['loss'] = loss.item()
        
        if dataset_idx > 0:
            dis_time = time.time()
           
            prev_model = prev_model.to(self.args.device)
            _,target = prev_model(src_nodes, dst_nodes, edges, edge_times, n_neighbors, dataset_idx,pre=True)
            with torch.no_grad():
                pre_emb_src,pre_emb_dst= prev_model.get_embeddings(src_nodes, dst_nodes, edges, edge_times, n_neighbors)

                emb_src,emb_dst,emb_src_neighbor = self.net.get_embeddings_with_neighbors(src_nodes, dst_nodes, edges, edge_times, n_neighbors) 
              
            if isinstance(target, tuple):
                target = target[0]  ## only src logits
            # logits = self.net(src_nodes, dst_nodes, edges, edge_times, n_neighbors, dataset_idx,return_logits=True)
            if isinstance(logits, tuple):
                logits = logits[0]  ## only src logits

         
            if target.size(1) > self.args.num_class_per_dataset:
                target = target[:,-self.args.num_class_per_dataset:]

            logits = logits[:,-self.args.num_class_per_dataset:]
     
            ## new classes in task t-1 vs new classes in tast t
            dis_loss = self.MultiClassCrossEntropy(logits, target, self.args.distill_loss['T'])  ## only src 
            dis_end =time.time()
          
            ## class distillation
            class_time = time.time()
                
            self.current_class_pro = self.get_current_prototype(emb_src,emb_dst,edges,dataset_idx)
            self.pre_current_class_pro = self.get_current_prototype(pre_emb_src,pre_emb_dst,edges,dataset_idx)
            # emb = torch.cat((emb_src, emb_dst), 0)  
            
            # class_loss = self.class_loss(emb_src,emb_dst,dataset_idx)

            class_end = time.time()
            
            del self.current_class_pro,self.pre_current_class_pro

            ## memory 
            mmd_time = time.time()

            with torch.no_grad():
                memory_src, memory_dst,memory_src_neighbor = self.net.get_embeddings_with_neighbors(self.memory.replay_data.src, self.memory.replay_data.dst,
                                                                  self.memory.replay_data.edge_idxs,self.memory.replay_data.timestamps, n_neighbors)
            # memory_emb = torch.cat((memory_src, memory_dst), 0)
            MMD_loss = LinearTimeMMDLoss()

            em_expanded = emb_src.unsqueeze(1)  # [300, 1, 300]
            src_tree = torch.cat((emb_src_neighbor, em_expanded), dim=1)  # [300, 6, 300]

            memory_emb_expand = memory_src.unsqueeze(1)
            memory_src_tree = torch.cat((memory_src_neighbor, memory_emb_expand), dim=1)
            mmd_loss = MMD_loss(src_tree, memory_src_tree) ## tree

            # mmd_loss = MMD_loss(emb_src, memory_src)

            mmd_end = time.time()
           
            # del emb,memory_emb

            # alpha = self.exponential_decay(self.args.distill_loss['lambda_dist'],dataset_idx)
            # beta = self.exponential_decay(self.args.mmd_loss,dataset_idx)
            alpha = self.args.distill_loss['lambda_dist']
            beta = self.args.mmd_loss
            
            # print(loss.item(),dis_loss.item(),class_loss.item(),mmd_loss.item())

            # loss = loss + alpha*(dis_loss + class_loss) + beta*mmd_loss
            
            # print(loss.item(),dis_loss.item(),mmd_loss.item())
            loss = loss + alpha*(dis_loss ) + beta*mmd_loss
            print('mmd_time',mmd_end-mmd_time)

            # wandb.log({'dis_time':dis_end-dis_time,'class_time':class_end-class_time,'mmd_time':mmd_end-mmd_time})
            # print(loss)

            # print('dis_time',dis_end-dis_time,'class_time',class_end-class_time,'mmd_time',mmd_end-mmd_time)

        
 
        data_dict['dist'] = dis_loss.item() if dataset_idx > 0 else 0
        # data_dict['class_dis'] = class_loss.item() if dataset_idx > 0 else 0
        data_dict['class_dis']  = 0
        data_dict['mmd'] = mmd_loss.item() if dataset_idx > 0 else 0
        
        data_dict['total_loss'] = loss.item()
        
        

        self.optimizer.zero_grad()
        loss.backward()
        # for name, param in self.net.named_parameters():
        #     if param.grad is not None:
        #         grad_norm = param.grad.data.norm(2)
        #         if grad_norm.item()>1:
        #             print('grad_norm',name,grad_norm)

        self.optimizer.step()
        
        return data_dict

    
    def get_current_prototype(self,emb_src,emb_dst,edges,task,current=True):
        
        train_class_src = self.get_class(self.src_label[edges],current)
        
        
        prototype_src = {}
        for key in train_class_src.keys():
            indices = train_class_src[key]
            assert all(idx < emb_src.size(0) for idx in indices), "Index out of range in emb_src"
            if len(indices) > 0:  # Ensure there are elements to avoid empty selection
                prototype_src[key] = torch.mean(emb_src[indices], dim=0,keepdim=True).to(self.args.device)

        return prototype_src
    


    def get_class(self,y_label,current):
        if current:
            labels = torch.unique(y_label)
        else:
            labels = np.unique(y_label)
        
        class_by_id = {str(label.item()): [] for label in labels}
        for i in range(len(y_label)):
            class_by_id[str(int(y_label[i].item()))].append(i)

        return class_by_id
    
    def get_prototype(self,data,task):
        current = False
        train_class_src = self.get_class(data.labels_src,current)

        with  torch.no_grad():
            emb_src,_ = self.net.get_embeddings(data.src, data.dst, data.edge_idxs, data.timestamps, self.args.num_neighbors)
           
        
        prototype_src = {}
        for key in train_class_src.keys():
            indices = train_class_src[key]
            assert all(idx < emb_src.size(0) for idx in indices), "Index out of range in emb_src"
            if len(indices) > 0:  # Ensure there are elements to avoid empty selection
                prototype_src[key] = torch.mean(emb_src[indices], dim=0,keepdim=True).to(self.args.device)
       
        for key in prototype_src.keys():
            if key in self.class_prototype:
                
                self.class_prototype[key] = torch.mean(self.class_prototype[key] + prototype_src[key],dim=0,keepdim=True)
                
            else:
                self.class_prototype[key] = prototype_src[key]
            
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

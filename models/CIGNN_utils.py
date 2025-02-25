
import numpy as np
import torch
from torch import device, nn
import torch.nn.functional as F
import torch.optim as optim
import math

from utils.utils import get_neighbor_finder

class IB(nn.Module): 
    def __init__(self, shape_x, shape_z, shape_y, label_src, label_dst, per_class, device, beta=0.3, dis_IB=False, ch_IB='m', n_task=6, lr=0.005):
        super(IB, self).__init__()
        self.label_src = label_src
        self.label_dst = label_dst
        self.per_class = per_class
        self.device = device
        self.beta = beta
        self.lr=lr
        self.dis_IB=dis_IB
        self.ch_IB=ch_IB
        self.n_task=n_task
        self.x_list=[]
        self.y_list=[]

        self.hidden1 = nn.Linear(shape_x, shape_z).to(self.device)
        self.hidden2 = nn.Linear(shape_z, shape_z).to(self.device)

        if self.ch_IB!='s':
            self.club = CLUBSample(x_dim=shape_z, y_dim=shape_y, hidden_size=shape_z, device=device)
        else:
            self.club = CLUBSample(x_dim=shape_z, y_dim=shape_y*self.n_task, hidden_size=shape_z, device=device)
            
        self.mine = MINE(x_dim=shape_x, y_dim=shape_z, hidden_size=shape_x, device=device)

        self.club_optimizer = optim.Adam(self.club.parameters(), lr=lr)
        self.mine_optimizer = optim.Adam(self.mine.parameters(), lr=lr)

        special_layers = torch.nn.ModuleList([self.club, self.mine])
        special_layers_params = list(map(id, special_layers.parameters()))
        base_params = filter(lambda p: id(p) not in special_layers_params, self.parameters())
        self.hidden_optimizer = optim.Adam(base_params, lr=lr)


    
    def forward(self, x, y, task_id, cur_id=0, ch='normal'):

        if self.club.p_mu2.weight.shape[0] < (task_id+1)*self.per_class and self.ch_IB!='s':
            if self.dis_IB:
                pass
            else:
                self.x_list=[]
                self.y_list=[]
                self.club.p_mu2=nn.Linear(self.club.p_mu2.weight.shape[1],(task_id+1)*self.per_class).to(self.device)
                self.club.p_logvar2=nn.Linear(self.club.p_logvar2.weight.shape[1],(task_id+1)*self.per_class).to(self.device)
                self.club.p_mu2.reset_parameters()
                self.club.p_logvar2.reset_parameters()

                self.club_optimizer = optim.Adam(self.club.parameters(), lr=self.lr)

                special_layers = torch.nn.ModuleList([self.club, self.mine])
                special_layers_params = list(map(id, special_layers.parameters()))
                base_params = filter(lambda p: id(p) not in special_layers_params, self.parameters())
                
                self.hidden_optimizer = optim.Adam(base_params, lr=self.lr)

                self.mine_optimizer = optim.Adam(self.mine.parameters(), lr=self.lr)
        
        else:
            pass

        if self.dis_IB:
            y = F.one_hot(y - cur_id*self.per_class, self.per_class)
        else:
            if self.ch_IB != 's':
                y = F.one_hot(y, (task_id+1)*self.per_class)
            else:
                y = F.one_hot(y, (self.n_task)*self.per_class)
            
        

        if len(self.x_list) < 1000 and ch=='normal':
            self.x_list.append(x)
            self.y_list.append(y)

        z = self.hidden1(x)
        z = F.relu(z)
        z = self.hidden2(z)
        
        return z

    def train_net(self, e):

        Obj = 0
        cnt = 0
        if len(self.x_list) > 0:
            for x, y in list(zip(self.x_list, self.y_list)):
                
                z = self.hidden1(x)
                z = F.relu(z)
                z = self.hidden2(z)
                I_zy = self.club.forward(z,y)
                I_xz = self.mine.forward(x,z)
                obj = I_zy - self.beta*I_xz 

                if abs(obj) > 1000:
                    obj = torch.clamp(obj, -1000, 1000)
                    pass
                else:
                    self.hidden_optimizer.zero_grad()
                    obj.backward()
                    self.hidden_optimizer.step()

                if e < 10:
                    z = self.hidden1(x)
                    z = F.relu(z)
                    z = self.hidden2(z)
                    loss_xz = self.mine.learning_loss(x,z)
                    if abs(loss_xz) > 1000:
                        loss_xz=torch.tensor([1])
                        pass
                    else:
                        self.mine_optimizer.zero_grad()
                        loss_xz.backward()
                        self.mine_optimizer.step()

                    z = self.hidden1(x)
                    z = F.relu(z)
                    z = self.hidden2(z)
                    loss_zy = self.club.learning_loss(z,y)
                    if abs(loss_zy) > 1000:
                        loss_zy=torch.tensor([1])
                        pass
                    else:
                        self.club_optimizer.zero_grad()
                        loss_zy.backward()
                        self.club_optimizer.step()

                Obj += obj.item()
                cnt += len(y)
        
        self.x_list=[]
        self.y_list=[]

        if cnt > 0:
            return Obj/cnt
        else: 
            return 0


class CLUBSample(nn.Module):  # Sampled version of the CLUB estimator
    def __init__(self, x_dim, y_dim, hidden_size, device):
        super(CLUBSample, self).__init__()
        
        self.device=device
        self.p_mu1=nn.Linear(x_dim, hidden_size).to(self.device)
        self.p_mu2=nn.Linear(hidden_size, y_dim).to(self.device)
        self.p_logvar1=nn.Linear(x_dim, hidden_size).to(self.device)
        self.p_logvar2=nn.Linear(hidden_size, y_dim).to(self.device)
        self.relu=nn.ReLU()
        self.tanh=nn.Tanh()

    def get_mu_logvar(self, x_samples):
        mu = self.p_mu1(x_samples)
        mu = self.relu(mu)
        mu = self.p_mu2(mu)
        
        logvar = self.p_logvar1(x_samples)
        logvar = self.relu(logvar)
        logvar = self.p_logvar2(logvar)
        logvar = self.tanh(logvar)

        return mu, logvar
     
        
    def loglikeli(self, x_samples, y_samples):
        mu, logvar = self.get_mu_logvar(x_samples)
        return (-(mu - y_samples)**2 /(logvar.exp() + 1e-6)-logvar).sum(dim=1).mean(dim=0)
    

    def forward(self, x_samples, y_samples):
        mu, logvar = self.get_mu_logvar(x_samples)
        
        sample_size = x_samples.shape[0]
        random_index = torch.randperm(sample_size).long()
        positive = - (mu - y_samples)**2 / (logvar.exp() + 1e-6)
        negative = - (mu - y_samples[random_index])**2 / (logvar.exp() + 1e-6)
        upper_bound = (positive.sum(dim = -1) - negative.sum(dim = -1)).mean()
        return upper_bound/2.

    def learning_loss(self, x_samples, y_samples):
        return - self.loglikeli(x_samples, y_samples)


class MINE(nn.Module):
    def __init__(self, x_dim, y_dim, hidden_size, device):
        super(MINE, self).__init__()
        self.device=device
        self.T_func = nn.Sequential(nn.Linear(x_dim + y_dim, hidden_size),
                                    nn.ReLU(),
                                    nn.Linear(hidden_size, 1)).to(self.device)
    
    def forward(self, x_samples, y_samples):  # samples have shape [sample_size, dim]
        # shuffle and concatenate
        sample_size = y_samples.shape[0]
        random_index = torch.randint(sample_size, (sample_size,)).long()

        y_shuffle = y_samples[random_index]

        T0 = self.T_func(torch.cat([x_samples,y_samples], dim = -1))
        T1 = self.T_func(torch.cat([x_samples,y_shuffle], dim = -1))

        # T0 = T0.tanh() * 10
        # T1 = T1.tanh() * 10
        # lower_bound = T0.mean() - torch.log(T1.exp().mean() + 1e-6)
        
        T1 = T1.view(T1.shape[0])
        T1 = torch.logsumexp(T1, dim=0) - math.log(T1.shape[0])
        lower_bound = T0.mean() - T1

        # compute the negative loss (maximise loss == minimise -loss)
        # lower_bound = torch.clamp(lower_bound, 0, 10)
        return lower_bound
    
    def learning_loss(self, x_samples, y_samples):
        return -self.forward(x_samples, y_samples)
    




class PGen(nn.Module):
    def __init__(self, per_class, node_init_dim, pmethod, node_label, node_feature, full_data, test_data, n_task, device):
        super(PGen, self).__init__()
        self.per_class = per_class
        self.node_init_dim = node_init_dim
        self.device = device
        self.mx_task = 0
        self.lr = 1e-3 # 1e-4
        self.pmethod = pmethod
        self.node_label = node_label
        self.node_feature = node_feature
        self.full_data = full_data
        self.test_data = test_data
        self.n_task = n_task
        self.dropout = nn.Dropout(p=0.5)
        if self.pmethod=='knn' or self.pmethod=='tknn':
            if self.pmethod == 'knn':
                self.n_neighbors = 50
            elif self.pmethod == 'tknn':
                self.n_neighbors = 50
            self.p_label = [0] * len(self.node_label)
            self.p_label[0] = -1
            self.p_label = torch.tensor(self.p_label)
            
            mem_label = [False] * len(self.node_label)
            mem_label = torch.tensor(mem_label)
            
            for c_task in range(self.n_task):
                if self.pmethod == 'knn':
                    test_neighbor_finder = get_neighbor_finder(full_data[c_task], False, mask=test_data[c_task])
                elif self.pmethod == 'tknn':
                    test_neighbor_finder = get_neighbor_finder(full_data[c_task], True, mask=test_data[c_task])
                bs = 300
                num_batch = math.ceil(len(full_data[c_task].src) / bs)
                for c_b in range(num_batch):
                    st_idx = c_b * bs
                    ed_idx = min((c_b + 1) * bs, len(full_data[c_task].src))
                    if ed_idx==st_idx:
                        break 
                    src_batch = full_data[c_task].src[st_idx:ed_idx]
                    dst_batch = full_data[c_task].dst[st_idx:ed_idx]
                    edge_batch = full_data[c_task].edge_idxs[st_idx:ed_idx]
                    timestamps_batch = full_data[c_task].timestamps[st_idx:ed_idx]         
                    for i, idxs in enumerate([src_batch, dst_batch]): 
                        if self.pmethod == 'knn':
                            f_timestamps = [1e15] * len(idxs)
                        elif self.pmethod == 'tknn':
                            f_timestamps = timestamps_batch
                        neighbors, _, n_times = test_neighbor_finder.get_temporal_neighbor(idxs, f_timestamps, self.n_neighbors)
                        neighbors = torch.from_numpy(neighbors).long().to(self.device)
                        bs = neighbors.shape[0]
                        neighbor_label = self.node_label[neighbors.flatten()]
                        neighbor_label = neighbor_label.view(bs, self.n_neighbors)
                        pred = []
                        cur_idx = -1
                        for cur_x in neighbor_label:
                            cur_idx += 1
                            cur_mask = (cur_x == c_task*self.per_class)
                            for cnt_c in range(1, self.per_class):
                                cur_mask = cur_mask | (cur_x == (c_task*self.per_class + cnt_c))
                            tmp_count = torch.bincount(cur_x[cur_mask] - c_task*self.per_class)
                            if len(tmp_count)==0:
                                tmp_count=torch.tensor([0])
                                mem_label[idxs[cur_idx]]=True

                            tmp_label = torch.argmax(tmp_count)
                            pred.append(tmp_label)
                        self.p_label[idxs] = torch.tensor(pred)
        
        elif self.pmethod=='mlp':
            self.W_m1 = nn.Parameter(torch.zeros((self.node_init_dim, 256)).to(self.device))
            self.W_m11 = nn.Parameter(torch.zeros((256, 128)).to(self.device))
            self.W_m2 = nn.Parameter(torch.zeros((128, per_class)).to(self.device))
            nn.init.xavier_normal_(self.W_m1)
            nn.init.xavier_normal_(self.W_m11)
            nn.init.xavier_normal_(self.W_m2)
                     
        elif self.pmethod=='nmlp':
            self.node_emb_dim = 100
            self.n_neighbors = 10
            self.use_feature = 'f'
            if self.use_feature == 'f':
                self.W_m1 = nn.Parameter(torch.zeros((self.node_init_dim + self.node_init_dim, 256)).to(self.device))
            else:
                self.W_m1 = nn.Parameter(torch.zeros((self.node_emb_dim + self.node_init_dim, 256)).to(self.device))
            self.W_m11 = nn.Parameter(torch.zeros((256, 128)).to(self.device))
            self.W_m2 = nn.Parameter(torch.zeros((128, per_class)).to(self.device))
            nn.init.xavier_normal_(self.W_m1)
            nn.init.xavier_normal_(self.W_m11)
            nn.init.xavier_normal_(self.W_m2)
                            
                                            
            
        if self.pmethod=='mlp' or self.pmethod=='nmlp':
        
            self.optimizer = optim.Adam(self.parameters(), lr=self.lr)
            self.criterion_list = nn.CrossEntropyLoss(reduction='none').to(self.device)

    def forward(self, node_feature, node_emb, src_idxs, dst_idxs, src_label, dst_label, task, neighbor_finder, ch='part'):
        
        src_feature=node_feature[src_idxs]
        dst_feature=node_feature[dst_idxs]
        
        
        if task > self.mx_task:
            self.mx_task = task
            if self.pmethod=='mlp' or self.pmethod=='nmlp':
                nn.init.xavier_normal_(self.W_m1)
                nn.init.xavier_normal_(self.W_m11)
                nn.init.xavier_normal_(self.W_m2)  
                self.optimizer = optim.Adam(self.parameters(), lr=self.lr)
            
        if ch=='part':
            src_task_mask = src_label == (task*self.per_class)
            dst_task_mask = dst_label == (task*self.per_class)
            for i in range(task*self.per_class, (task+1)*self.per_class):
                src_task_mask |= (src_label == i)
                dst_task_mask |= (dst_label == i)

            cur_label_src = src_label[src_task_mask]
            cur_label_dst = dst_label[dst_task_mask]
            
        
        
        if self.pmethod=='knn' or self.pmethod=='tknn':
            
            if ch=='part':
                src_idxs=src_idxs[src_task_mask.detach().cpu()]
                dst_idxs=dst_idxs[dst_task_mask.detach().cpu()]
            
                    
            for i, idxs in enumerate([src_idxs, dst_idxs]): # traverse src and dst in turns
                pred = F.one_hot(self.p_label[idxs], self.per_class)
                pred = pred.float().to(self.device)
                if i==0:
                    src_logits2 = pred
                else:
                    dst_logits2 = pred    
              
        
        elif self.pmethod=='mlp':

            if ch=='part':

                src_feature = src_feature[src_task_mask]
                dst_feature = dst_feature[dst_task_mask]

            pre_src_logits2=torch.matmul(src_feature, self.W_m1)
            pre_dst_logits2=torch.matmul(dst_feature, self.W_m1)

            pre_src_logits2=F.relu(pre_src_logits2)
            pre_src_logits2=torch.matmul(pre_src_logits2, self.W_m11)
            pre_dst_logits2=F.relu(pre_dst_logits2)
            pre_dst_logits2=torch.matmul(pre_dst_logits2, self.W_m11)

            pre_src_logits2 = F.relu(pre_src_logits2)
            pre_dst_logits2 = F.relu(pre_dst_logits2)
            pre_src_logits2 = self.dropout(pre_src_logits2)
            pre_dst_logits2 = self.dropout(pre_dst_logits2)

            src_logits2 = torch.matmul(pre_src_logits2, self.W_m2)
            dst_logits2 = torch.matmul(pre_dst_logits2, self.W_m2)
            

                    
        elif self.pmethod=='nmlp':
            
            if ch=='part':
                src_idxs=src_idxs[src_task_mask.detach().cpu()]
                dst_idxs=dst_idxs[dst_task_mask.detach().cpu()]
                    
            message = []
            for i, idxs in enumerate([src_idxs, dst_idxs]): # traverse src and dst in turns
                f_timestamps = [1e15] * len(idxs)
                neighbors, _, n_times = neighbor_finder.get_temporal_neighbor(idxs, f_timestamps, self.n_neighbors)
                neighbors = torch.from_numpy(neighbors).long().to(self.device)
                bs = neighbors.shape[0]
                if self.use_feature=='f':
                    neighbor_emb = node_feature[neighbors.flatten()]
                    neighbor_emb = neighbor_emb.view(bs, self.n_neighbors, self.node_init_dim)
                else:
                    neighbor_emb = node_emb[neighbors.flatten()]
                    neighbor_emb = neighbor_emb.view(bs, self.n_neighbors, self.node_emb_dim)
                
                h = neighbor_emb.mean(dim=1)
                message.append(h)
                
            pre_src_logits = torch.matmul(torch.cat((message[0], node_feature[src_idxs]),dim=1), self.W_m1)
            pre_dst_logits = torch.matmul(torch.cat((message[1], node_feature[dst_idxs]),dim=1), self.W_m1)
            pre_src_logits = F.relu(pre_src_logits)
            pre_dst_logits = F.relu(pre_dst_logits)
            pre_src_logits = self.dropout(pre_src_logits)
            pre_dst_logits = self.dropout(pre_dst_logits)
            src_logits2 = torch.matmul(pre_src_logits, self.W_m11)
            dst_logits2 = torch.matmul(pre_dst_logits, self.W_m11)
            
            src_logits2 = torch.matmul(src_logits2, self.W_m2)
            dst_logits2 = torch.matmul(dst_logits2, self.W_m2)
        

        
        if self.pmethod=='mlp' or self.pmethod=='nmlp':

            if ch=='part':
                loss_s2 = self.criterion_list(src_logits2, cur_label_src - task*self.per_class)
                loss_d2 = self.criterion_list(dst_logits2, cur_label_dst - task*self.per_class)
                return src_logits2, dst_logits2, loss_s2, loss_d2
            else:
                return src_logits2, dst_logits2
            
        else:
            
            if ch=='part':
                return src_logits2, dst_logits2, torch.tensor([0.]), torch.tensor([0.])
            else:
                return src_logits2, dst_logits2
            
        
        

    def train_net(self, loss, e):
        
        if self.pmethod=='mlp' or self.pmethod=='nmlp': 
            if (e+1)%100 == 0:
                self.optimizer.param_groups[0]['lr']/=2

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
        else:
            pass
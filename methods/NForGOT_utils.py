import torch
from torch import nn
import numpy as np


class Data:
    def __init__(
        self, src, dst, timestamps, edge_idxs, labels_src, labels_dst, induct_nodes=None
    ):
        self.src = src
        self.dst = dst

        self.timestamps = timestamps
        self.edge_idxs = edge_idxs
        self.labels_src = labels_src
        self.labels_dst = labels_dst

        self.n_interactions = len(src)
        self.unique_nodes = set(src) | set(dst)
        self.n_unique_nodes = len(self.unique_nodes)
        self.induct_nodes = induct_nodes

    def add_data(self, x):
        if x is not None:
            self.src = np.concatenate((self.src, x.src))
            self.dst = np.concatenate((self.dst, x.dst))

            self.timestamps = np.concatenate((self.timestamps, x.timestamps))
            self.edge_idxs = np.concatenate((self.edge_idxs, x.edge_idxs))
            self.labels_src = np.concatenate((self.labels_src, x.labels_src))
            self.labels_dst = np.concatenate((self.labels_dst, x.labels_dst))

            self.n_interactions = len(self.src)
            self.unique_nodes = set(self.src) | set(self.dst)
            self.n_unique_nodes = len(self.unique_nodes)

class Memory(nn.Module):
    def __init__(self):
        super(Memory, self).__init__()
        self.memory = [] ## sampled data from each previous tasks
        self.total_memory = None  ## sampled data from all previous tasks
        self.replay_data = None ## final sampling
        self.curr_memory = None
       
        # self.memory_size = memory_size

    def update_memory(self,sampled_idx,data):
        new_memory= Data(data.src[sampled_idx], data.dst[sampled_idx], data.timestamps[sampled_idx], data.edge_idxs[sampled_idx], data.labels_src[sampled_idx], data.labels_dst[sampled_idx])

        self.memory.append(new_memory) 
        self.curr_memory = new_memory
        if self.total_memory is None:
            self.total_memory = new_memory
        else:
            self.total_memory.add_data(new_memory)

    def get_data(self, size, mode='random'):
        if size > len(self.total_memory.src):
            size = len(self.total_memory.src)
        if mode == 'random':
             ## radnomly sample data from the total memory ( batchsize per task and samlpe batchsize from all previous tasks)
            idx = np.random.choice(len(self.total_memory.src), size, replace=False)
            self.replay_data = Data(self.total_memory.src[idx], self.total_memory.dst[idx],self.total_memory.timestamps[idx],self.total_memory.edge_idxs[idx],
                                 self.total_memory.labels_src[idx], self.total_memory.labels_dst[idx])
    def update_data(self, data):
        if self.replay_data is not None:
            data.add_data(self.replay_data) 
        return data
    
    def pre_batch(self, data,st_idx,ed_idx):
        if ed_idx > len(data.src):
            idx = np.random.choice(len(data.src), ed_idx-st_idx+1, replace=False)
            pre_batch_data = Data(data.src[idx], data.dst[idx], data.timestamps[idx], data.edge_idxs[idx], data.labels_src[idx], data.labels_dst[idx])
        else:
            pre_batch_data = Data(data.src[st_idx:ed_idx], data.dst[st_idx:ed_idx], data.timestamps[st_idx:ed_idx], data.edge_idxs[st_idx:ed_idx], data.labels_src[st_idx:ed_idx], data.labels_dst[st_idx:ed_idx])
        return pre_batch_data
    
    

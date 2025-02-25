import torch
from torch import nn
import numpy as np

import math
from models.Backbone import TemporalGNNClassifier

# The following code is to initialize the class for finetune, which is a vanilla baseline in continual learning.
# Please generate a template code for me.

class SubGraph(nn.Module):
    def __init__(self, args, neighbor_finder, node_features, edge_features, src_label, dst_label):
        super(SubGraph, self).__init__()
        self.args = args

        if args.supervision == 'supervised':
            self.model = TemporalGNNClassifier(args, neighbor_finder, node_features, edge_features, src_label, dst_label)
        elif self.args.supervision == 'semi-supervised':
            return NotImplementedError
        
        self.memory = Memory()
        
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=args.lr)
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=100, gamma=0.5)

    def forward(self, src_nodes, dst_nodes, edges, edge_times, n_neighbors, dataset_idx=None):
        self.model.detach_memory()

        if self.args.task == 'nodecls':
            return self.forward_nodecls(src_nodes, dst_nodes, edges, edge_times, n_neighbors, dataset_idx)
        elif self.args.task == 'linkpred':
            return self.forward_linkpred(src_nodes, dst_nodes, edges, edge_times, n_neighbors, dataset_idx)

    def forward_linkpred(self, src_nodes, dst_nodes, edges, edge_times, n_neighbors, dataset_idx):
        return
    
    def forward_nodecls(self, src_nodes, dst_nodes, edges, edge_times, n_neighbors, dataset_idx=None):
        
        if self.args.memory_replay and dataset_idx > 0:
            memory_src_nodes, memory_dst_nodes, memory_edge_idxs, memory_timestamps = self.memory.get_data(self.args.batch_size, mode=self.args.replay_select_mode)
            
        data_dict = {}

        if self.args.supervision == 'supervised':
            loss = self.model(src_nodes, dst_nodes, edges, edge_times, n_neighbors, dataset_idx)
            data_dict['loss'] = loss.item()

            if self.args.memory_replay and dataset_idx > 0:
                memory_loss = self.model(memory_src_nodes, memory_dst_nodes, memory_edge_idxs, memory_timestamps, n_neighbors, dataset_idx)
                data_dict['memory_loss'] = memory_loss.item()
                loss = loss + memory_loss * self.args.memory_replay_weight

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
        elif self.args.supervision == 'semi-supervised':
            return NotImplementedError
        
        return data_dict

    def set_neighbor_finder(self, neighbor_finder):
        self.model.set_neighbor_finder(neighbor_finder)

    def detach_memory(self):
        self.model.detach_memory()

    def end_epoch(self):
        self.scheduler.step()

    def get_logits(self, src_nodes, dst_nodes, edges, edge_times, n_neighbors, dataset_idx):
        src_logits, dst_logits = self.model(src_nodes, dst_nodes, edges, edge_times, n_neighbors, dataset_idx, return_logits=True)
        return src_logits, dst_logits
    
    def end_dataset(self, train_data, args):
        if not args.memory_replay:
            return
        elif args.select_mode == 'random':
            sampled_idx = np.random.choice(len(train_data.src), args.memory_size, replace=False)
            cur_memory = Data(train_data.src[sampled_idx], train_data.dst[sampled_idx], train_data.timestamps[sampled_idx], train_data.edge_idxs[sampled_idx], train_data.labels_src[sampled_idx], train_data.labels_dst[sampled_idx])
            self.memory.update_memory(cur_memory)

        elif args.select_mode == 'mean_emb':
            src_emb_bank = []
            dst_emb_bank = []
            with torch.no_grad():
                num_batch = math.ceil(len(train_data.src) / args.batch_size)

                for i in range(num_batch):

                    st_idx = i * args.batch_size
                    ed_idx = min((i + 1) * args.batch_size, len(train_data.src))

                    src_batch = train_data.src[st_idx:ed_idx]
                    dst_batch = train_data.dst[st_idx:ed_idx]
                    edge_batch = train_data.edge_idxs[st_idx:ed_idx]
                    timestamp_batch = train_data.timestamps[st_idx:ed_idx]

                    src_embeddings, dst_embeddings = self.model.get_embeddings(src_batch, dst_batch, edge_batch, timestamp_batch, args.num_neighbors)
                    src_emb_bank.append(src_embeddings)
                    dst_emb_bank.append(dst_embeddings)
                
                src_emb_bank = torch.cat(src_emb_bank, dim=0)
                dst_emb_bank = torch.cat(dst_emb_bank, dim=0)
            
            seen_task = len(self.memory.memory)
            cur_memory = None
            for c in range(seen_task * self.args.num_class_per_dataset, (seen_task + 1) * self.args.num_class_per_dataset):
                task_mp_src = torch.tensor([False for i in range(len(src_emb_bank))])
                task_mp_dst = torch.tensor([False for i in range(len(dst_emb_bank))])  
                task_mp_src = task_mp_src | (torch.tensor(train_data.labels_src) == c)
                task_mp_dst = task_mp_dst | (torch.tensor(train_data.labels_dst) == c)

                task_mp = task_mp_src | task_mp_dst

                total_emb_bank = torch.cat([src_emb_bank[task_mp_src], dst_emb_bank[task_mp_dst]], dim=0)
                mean_emb = torch.mean(total_emb_bank)

                dist_src = -torch.nn.CosineSimilarity()(src_emb_bank[task_mp], mean_emb)
                dist_dst = -torch.nn.CosineSimilarity()(dst_emb_bank[task_mp], mean_emb)

                total_dist = dist_src + dist_dst

                memory_size_class = int(args.memory_size / self.args.num_class_per_dataset)
                memory_size_class = min(memory_size_class, len(total_dist))
                _, sampled_idx_class = torch.sort(total_dist, stable=True)
                sampled_idx_class = sampled_idx_class[:memory_size_class].cpu()

                if cur_memory == None:
                    cur_memory = Data(train_data.src[task_mp][sampled_idx_class], train_data.dst[task_mp][sampled_idx_class], \
                                      train_data.timestamps[task_mp][sampled_idx_class], train_data.edge_idxs[task_mp][sampled_idx_class], \
                                      train_data.labels_src[task_mp][sampled_idx_class], train_data.labels_dst[task_mp][sampled_idx_class])
                else:
                    cur_memory.add_data(Data(train_data.src[task_mp][sampled_idx_class], train_data.dst[task_mp][sampled_idx_class], \
                                      train_data.timestamps[task_mp][sampled_idx_class], train_data.edge_idxs[task_mp][sampled_idx_class], \
                                      train_data.labels_src[task_mp][sampled_idx_class], train_data.labels_dst[task_mp][sampled_idx_class]))

            self.memory.update_memory(cur_memory)
            
        elif self.args.select_mode == 'mean_pred':
            src_pred_bank = []
            dst_pred_bank = []
            with torch.no_grad():
                num_batch = math.ceil(len(train_data.src) / args.batch_size)

                for i in range(num_batch):

                    st_idx = i * args.batch_size
                    ed_idx = min((i + 1) * args.batch_size, len(train_data.src))

                    src_batch = train_data.src[st_idx:ed_idx]
                    dst_batch = train_data.dst[st_idx:ed_idx]
                    edge_batch = train_data.edge_idxs[st_idx:ed_idx]
                    timestamp_batch = train_data.timestamps[st_idx:ed_idx]

                    src_preds, dst_preds = self.model(src_batch, dst_batch, edge_batch, timestamp_batch, args.num_neighbors, return_logits=True)
                    src_pred_bank.append(src_preds)
                    dst_pred_bank.append(dst_preds)
                
                src_pred_bank = torch.cat(src_pred_bank, dim=0)
                dst_pred_bank = torch.cat(dst_pred_bank, dim=0)
            
            zero_preds = torch.zeros(src_pred_bank.shape).to(self.args.device)
            seen_task = len(self.memory.memory)
            cur_memory = None
            for c in range(seen_task * self.args.num_class_per_dataset, (seen_task + 1) * self.args.num_class_per_dataset):
                task_mp_src = torch.tensor([False for i in range(len(src_pred_bank))])
                task_mp_dst = torch.tensor([False for i in range(len(dst_pred_bank))])  
                task_mp_src = task_mp_src | (torch.tensor(train_data.labels_src) == c)
                task_mp_dst = task_mp_dst | (torch.tensor(train_data.labels_dst) == c)

                task_mp = task_mp_src | task_mp_dst
                # task_mp_and = task_mp_src & task_mp_dst
                # task_mp_src_and = task_mp_src & (~task_mp_dst)
                # task_mp_dst_and = task_mp_dst & (~task_mp_src)
                
                temp_src_pred_bank = torch.zeros(src_pred_bank.shape).to(self.args.device)
                temp_src_pred_bank[task_mp_src] = src_pred_bank[task_mp_src]

                temp_dst_pred_bank = torch.zeros(dst_pred_bank.shape).to(self.args.device)
                temp_dst_pred_bank[task_mp_dst] = dst_pred_bank[task_mp_dst]

                mean_pred_event = (temp_src_pred_bank[task_mp] + temp_dst_pred_bank[task_mp]) / 2

                memory_size_class = int(args.memory_size / self.args.num_class_per_dataset)
                memory_size_class = min(memory_size_class, len(mean_pred_event))
                _, sampled_idx_class = torch.sort(mean_pred_event, stable=True, descending=True)
                sampled_idx_class = sampled_idx_class[:memory_size_class].cpu()

                print(sampled_idx_class.shape)

                if cur_memory == None:
                    cur_memory = Data(train_data.src[task_mp][sampled_idx_class], train_data.dst[task_mp][sampled_idx_class], \
                                      train_data.timestamps[task_mp][sampled_idx_class], train_data.edge_idxs[task_mp][sampled_idx_class], \
                                      train_data.labels_src[task_mp][sampled_idx_class], train_data.labels_dst[task_mp][sampled_idx_class])
                else:
                    cur_memory.add_data(Data(train_data.src[task_mp][sampled_idx_class], train_data.dst[task_mp][sampled_idx_class], \
                                      train_data.timestamps[task_mp][sampled_idx_class], train_data.edge_idxs[task_mp][sampled_idx_class], \
                                      train_data.labels_src[task_mp][sampled_idx_class], train_data.labels_dst[task_mp][sampled_idx_class]))

            self.memory.update_memory(cur_memory)

        elif self.args.select_mode == 'explainer':
            from SubGraph_utils import PGExplainerExt

            explainer = PGExplainerExt(self.model.base_model, self.args.model, self.args.explainer, self.args.dataset, self.args.node_init_dim, self.args, \
                                       train_data, self.args.explanation_level, self.args.device, self.args.verbose, self.args.results_dir, self.args.debug_mode, \
                                        self.args.explainer_train_epochs, self.args.explainer_ckpt_dir, self.args.explainer_reg_coefs, self.args.explainer_batch_size, self.args.explainer_lr)


            explainer._train()


            # train the explainer

            

# self, model, model_name: str, explainer_name: str, dataset_name: str, input_dim: int, args,
#                  all_events: DataFrame,  explanation_level: str, device, verbose: bool = True, results_dir = None, debug_mode=True,
#                  # specific params for PGExplainerExt
#                  train_epochs: int = 50, explainer_ckpt_dir = None, reg_coefs = None, batch_size = 64, lr=1e-4


    def get_acc(self, x, y):
        output = self.model(x)
        _, pred = torch.max(output, 1)
        correct = (pred == y).sum().item()
        return correct

    def reset_graph(self):
        return

    def back_up_memory(self):
        return

    def restore_memory(self, back_up):
        return

    def get_parameters(self):
        return self.model.parameters()

    def get_optimizer(self):
        return self.optimizer

    def get_scheduler(self):
        return self.scheduler

    def get_model(self):
        return self.model


class Memory(nn.Module):
    def __init__(self):
        super(Memory, self).__init__()
        self.memory = []
        self.total_memory = None
        # self.memory_size = memory_size

    
    def update_memory(self, new_memory):
        if len(self.memory) == 0:
            self.memory.append(new_memory)
            self.total_memory = new_memory
        else:
            self.memory.append(new_memory)
            self.total_memory.add_data(new_memory)
        

    def get_data(self, size, mode='random'):
        if size > len(self.total_memory.src):
            size = len(self.total_memory.src)
        if mode == 'random':
            idx = np.random.choice(len(self.total_memory.src), size, replace=False)
            return self.total_memory.src[idx], self.total_memory.dst[idx], self.total_memory.edge_idxs[idx], self.total_memory.timestamps[idx]
    

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
        self.src = np.concatenate((self.src, x.src))
        self.dst = np.concatenate((self.dst, x.dst))

        self.timestamps = np.concatenate((self.timestamps, x.timestamps))
        self.edge_idxs = np.concatenate((self.edge_idxs, x.edge_idxs))
        self.labels_src = np.concatenate((self.labels_src, x.labels_src))
        self.labels_dst = np.concatenate((self.labels_dst, x.labels_dst))

        self.n_interactions = len(self.src)
        self.unique_nodes = set(self.src) | set(self.dst)
        self.n_unique_nodes = len(self.unique_nodes)
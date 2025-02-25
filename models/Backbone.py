# This is a class that takes the models of temporal gnns of this folder and add a classification head on top of it.
# give me a template code for this class.

import torch
import torch.nn as nn
from typing import List

from copy import deepcopy

from .TGAT import TGAN
from .TGN import TGN
# from .CIGNN import CIGNN
from .TGAT_dyglib import TGAT
from .TCL import TCL
from .DyGFormer import DyGFormer

def kaiming_normal_init(m):
    
    if isinstance(m, torch.nn.Conv2d):
        torch.nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
    
    elif isinstance(m, torch.nn.Linear):
        torch.nn.init.kaiming_normal_(m.weight, nonlinearity='sigmoid')
  

def get_base_model(args, neighbor_finder, node_features, edge_features):

    if args.model == 'TGAT':
        # return TGAN(neighbor_finder, node_features, edge_features, device=args.device,
        #         attn_mode='prod', use_time='time', agg_method='attn', node_dim=None, time_dim=args.time_dim,
        #         num_layers=args.num_layer, n_head=args.num_attn_heads, null_idx=0, drop_out=args.dropout, seq_len=None)
        return TGAT(node_raw_features=node_features, edge_raw_features=edge_features, neighbor_sampler=neighbor_finder,
                    time_feat_dim=args.time_dim, num_layers=args.num_layer, num_heads=args.num_attn_heads, dropout=args.dropout, device=args.device)
    elif args.model == 'TGN':
        return TGN(neighbor_finder=neighbor_finder, node_features=node_features, edge_features=edge_features, device=args.device, n_layers=args.num_layer,
                            n_heads=args.num_attn_heads, dropout=args.dropout, use_memory=True, forbidden_memory_update=False,
                            memory_update_at_start=True, 
                            message_dimension=128, memory_dimension=128, embedding_module_type="graph_attention",
                            message_function="identity",
                            mean_time_shift_src=args.time_shift['mean_time_shift_src'], std_time_shift_src=args.time_shift['std_time_shift_src'],
                            mean_time_shift_dst=args.time_shift['mean_time_shift_dst'], std_time_shift_dst=args.time_shift['std_time_shift_dst'], 
                            n_neighbors=args.num_neighbors, aggregator_type="last", memory_updater_type="gru",
                            use_destination_embedding_in_message=True,
                            use_source_embedding_in_message=True,
                            dyrep=False)
   
    elif args.model == 'DyGFormer':
         return DyGFormer(node_raw_features=node_features, edge_raw_features=edge_features, neighbor_sampler=neighbor_finder,
                                         time_feat_dim=args.time_dim, channel_embedding_dim=args.channel_embedding_dim, patch_size=args.patch_size,
                                         num_layers=args.num_layer, num_heads=args.num_attn_heads, dropout=args.dropout,
                                         max_input_sequence_length=args.max_input_sequence_length, device=args.device)
    elif args.model == 'TCL':
        return TCL(node_raw_features=node_features, edge_raw_features=edge_features, neighbor_sampler=neighbor_finder,
                 time_feat_dim=args.time_dim, num_layers=args.num_layer, num_heads=args.num_attn_heads, num_depths=args.num_neighbors + 1,dropout=args.dropout, device=args.device)
    

class IncrementalClassifier(nn.Module):
    def __init__(self, input_dim: int, num_classes: int, device, masking = False):
        super().__init__()
        self.init_class = num_classes
        self.masking = masking
        self.classifier = nn.Linear(input_dim, num_classes)  # 修改为类别数
        au_int = torch.zeros(num_classes, dtype=torch.int8)
        self.register_buffer("active_units", au_int)
        self.mask_value = -1000
        self.device = device
        # self.act = nn.ReLU()
        # self.dropout = nn.Dropout(dropout)

    @torch.no_grad()
    def adaptation(self,tasks):
        """If `dataset` contains unseen classes the classifier is expanded.

        :param experience: data from the current experience.
        :return:
        """
        device = self.device
        in_features = self.classifier.in_features
        old_nclasses = self.classifier.out_features
        print('old_nclasses',old_nclasses)
        new_nclasses = self.init_class * (tasks + 1)
        curr_classes = list(range(self.init_class * tasks, self.init_class * (tasks + 1)))

        print('new_nclasses',new_nclasses)
        print('curr_classes',curr_classes)
        # update active_units mask
        if self.masking:
            if old_nclasses != new_nclasses:  # expand active_units mask
                old_act_units = self.active_units
                self.active_units = torch.zeros(
                    new_nclasses, dtype=torch.int8, device=device
                )
                self.active_units[: old_act_units.shape[0]] = old_act_units
            # update with new active classes
            if self.training:
                self.active_units[list(curr_classes)] = 1

        # update classifier weights
        if old_nclasses == new_nclasses:
            return
 
        
        old_w, old_b = self.classifier.weight.clone().detach(), self.classifier.bias.clone().detach()
        self.classifier = torch.nn.Linear(in_features, new_nclasses).to(device)
        kaiming_normal_init(self.classifier.weight[old_nclasses:])  #  use or not use  initialization
        self.classifier.weight[:old_nclasses].copy_(old_w)    # Detach from the current computation graph
        self.classifier.bias[:old_nclasses].copy_(old_b)
       
        # old_w, old_b = self.classifier.weight, self.classifier.bias
        # self.classifier = torch.nn.Linear(in_features, new_nclasses).to(device)
        # self.classifier.weight[:old_nclasses] = old_w
        # self.classifier.bias[:old_nclasses] = old_b
        
    def forward(self, x: torch.Tensor,task=None):
        if task:
            curr_classes = list(range(self.init_class * task, self.init_class * (task + 1)))
            self.active_units[list(curr_classes)] = 1
            
        out = self.classifier(x)
        if self.masking:
            mask = torch.logical_not(self.active_units)
            out = out.masked_fill(mask=mask, value=self.mask_value)
        return out

class TemporalGNNClassifier(nn.Module):
    def __init__(self, args, neighbor_finder, node_features, edge_features, src_label, dst_label):
        super(TemporalGNNClassifier, self).__init__()
        self.args = args

        
        self.src_label = src_label
        self.dst_label = dst_label
        self.node_features = node_features
        self.base_model = get_base_model(args, neighbor_finder, node_features, edge_features)
        self.multihead = args.multihead

        if self.multihead:
            self.num_heads = args.num_datasets

        self.num_class_per_dataset = args.num_class_per_dataset

        self.criterion = nn.CrossEntropyLoss()

        self.features = nn.Sequential(
            nn.Linear(node_features.shape[1], args.head_hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(args.dropout),
        )
        # self.classification = MLPClassifier_multi_class(input_dim=node_features.shape[1] ,num_classes=args.num_class_per_dataset)
        self.incrementalclassifier = False
        if self.incrementalclassifier:
            self.classification = IncrementalClassifier(input_dim=args.head_hidden_dim ,num_classes=args.num_class_per_dataset,device=args.device,masking =True)
        elif self.multihead:
            self.classification = nn.ModuleList([nn.Linear(args.head_hidden_dim, args.num_class_per_dataset) for _ in range(args.num_datasets)])
            if self.args.dataset !='amazon':
                self.classification.apply(kaiming_normal_init)   #### 
        else:
            self.classification = nn.Linear(args.head_hidden_dim, args.num_class_per_dataset)

    def get_embeddings(self, src_nodes, dst_nodes, edges, edge_times, n_neighbors):
        return self.base_model(src_nodes, dst_nodes, edges, edge_times, n_neighbors)
        # return self.base_model(src_nodes, dst_nodes, edges, edge_times, n_neighbors)
    def get_embeddings_with_neighbors(self, src_nodes, dst_nodes, edges, edge_times, n_neighbors):
        return self.base_model.forward_with_neighbors(src_nodes, dst_nodes, edges, edge_times, n_neighbors)

    def forward(self, src_nodes, dst_nodes, edges, edge_times, n_neighbors, dataset_idx=None, return_logits=False,pre=None, candidate_weights_dict=None):
       
        # self.src_label = torch.tensor(self.src_label).to(self.args.device)
        # self.dst_label = torch.tensor(self.dst_label).to(self.args.device)
        cur_label_src = deepcopy(self.src_label[edges])
        cur_label_dst = deepcopy(self.dst_label[edges])

        src_embeddings, dst_embeddings = self.base_model(src_nodes, dst_nodes, edges, edge_times, n_neighbors)
        src_out = self.features(src_embeddings)
        dst_out = self.features(dst_embeddings)

        if self.incrementalclassifier:
            # self.classification.adaptation(dataset_idx)
            if pre:
                src_ds_mask = (cur_label_src >=  self.num_class_per_dataset) & (cur_label_src < (dataset_idx + 1) * self.num_class_per_dataset)
                cur_label_src[src_ds_mask] = cur_label_src[src_ds_mask] -  self.num_class_per_dataset

  
            src_preds = self.classification(src_out,dataset_idx)
            dst_preds = self.classification(dst_out,dataset_idx)

            labels = torch.from_numpy(cur_label_src).long().to(src_preds.device)
            loss = self.criterion(src_preds, labels)
        
        elif self.multihead:

            src_preds = torch.zeros((len(src_out), self.num_class_per_dataset)).to(self.args.device)
            dst_preds = torch.zeros((len(dst_out), self.num_class_per_dataset)).to(self.args.device)
            
            for ds_id in range(self.num_heads):
                src_ds_mask = (cur_label_src >= ds_id * self.num_class_per_dataset) & (cur_label_src < (ds_id + 1) * self.num_class_per_dataset)
                dst_ds_mask = (cur_label_dst >= ds_id * self.num_class_per_dataset) & (cur_label_dst < (ds_id + 1) * self.num_class_per_dataset)
                
                if src_ds_mask.sum() == 0 and dst_ds_mask.sum() == 0:
                    continue

                # Convert the labels to be within the range of the current dataset
                cur_label_src[src_ds_mask] = cur_label_src[src_ds_mask] - ds_id * self.num_class_per_dataset
                cur_label_dst[dst_ds_mask] = cur_label_dst[dst_ds_mask] - ds_id * self.num_class_per_dataset

                
                
                src_preds[src_ds_mask] += self.classification[ds_id](src_out[src_ds_mask])
                dst_preds[dst_ds_mask] += self.classification[ds_id](dst_out[dst_ds_mask])
         

            # Ensure labels are within valid range

            cur_label_src= torch.tensor(cur_label_src).to(self.args.device)
            cur_label_dst = torch.tensor(cur_label_dst).to(self.args.device)
            assert torch.all(cur_label_src >= 0) and torch.all(cur_label_src < src_preds.shape[1]), "cur_label_src contains invalid labels."
            assert torch.all(cur_label_dst >= 0) and torch.all(cur_label_dst < dst_preds.shape[1]), "cur_label_dst contains invalid labels."

            
            loss_src = self.criterion(src_preds, cur_label_src)

            loss = loss_src 
        else:
            a = 0
            # src_ds_mask = (cur_label_src >= dataset_idx * self.num_class_per_dataset) & (cur_label_src < (dataset_idx + 1) * self.num_class_per_dataset)
            # dst_ds_mask = (cur_label_src >= dataset_idx * self.num_class_per_dataset) & (cur_label_src < (dataset_idx + 1) * self.num_class_per_dataset)

            # # cur_label_src = cur_label_src[src_ds_mask]
            # # cur_label_dst = cur_label_src[dst_ds_mask]

            # cur_label_src[src_ds_mask] = cur_label_src[src_ds_mask] - dataset_idx * self.num_class_per_dataset
            
            # # src_out = self.features(src_embeddings)
            # # dst_out = self.features(dst_embeddings)
            # src_preds = torch.zeros((len(src_out), self.num_class_per_dataset)).to(self.args.device)
            # dst_preds = torch.zeros((len(dst_out), self.num_class_per_dataset)).to(self.args.device)

            # src_preds[src_ds_mask] = self.classification(self.features(src_embeddings[src_ds_mask]))
            # dst_preds[dst_ds_mask] = self.classification(self.features(dst_embeddings[dst_ds_mask]))

            # labels = torch.from_numpy(cur_label_src).long().to(src_preds.device)
            # loss = self.criterion(src_preds[src_ds_mask], labels[src_ds_mask])

        if return_logits:
            return src_preds, dst_preds, loss
        
        return loss, src_preds
    
    def adaptation(self,tasks):
        self.classification.adaptation(tasks)
    
    def set_neighbor_finder(self, neighbor_finder):
        self.base_model.set_neighbor_finder(neighbor_finder)

    def reset_graph(self):
        self.base_model.reset_graph()

    def detach_memory(self):
        # if self.args.method == 'OTGNet':
        #     self.base_model.detach_memory()
        # else:
        return
           

class MLPClassifier_multi_class(nn.Module):
    def __init__(self, input_dim: int, num_classes: int, dropout: float = 0.1):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, 80)
        self.fc2 = nn.Linear(80, 10)
        self.fc3 = nn.Linear(10, num_classes)  # 修改为类别数
        self.act = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor):
        x = self.dropout(self.act(self.fc1(x)))
        x = self.dropout(self.act(self.fc2(x)))
        # 不应用Softmax; CrossEntropyLoss 会处理
        return self.fc3(x)


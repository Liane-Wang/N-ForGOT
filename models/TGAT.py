import logging

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from models.TGAT_utils import *

class TGAN(torch.nn.Module):
    def __init__(self, ngh_finder, n_feat, e_feat, device,
                 attn_mode='prod', use_time='time', agg_method='attn', node_dim=None, time_dim=None,
                 num_layers=3, n_head=4, null_idx=0, drop_out=0.1, seq_len=None):
        super(TGAN, self).__init__()
        
        self.device = device
        self.num_layers = num_layers 
        self.ngh_finder = ngh_finder
        self.null_idx = null_idx
        # self.logger = logging.getLogger(__name__)
        self.n_feat_th = torch.nn.Parameter(torch.from_numpy(n_feat.astype(np.float32)))
        self.e_feat_th = torch.nn.Parameter(torch.from_numpy(e_feat.astype(np.float32)))
        # self.edge_raw_features = torch.nn.Embedding.from_pretrained(self.e_feat_th, padding_idx=0, freeze=True)
        # self.node_raw_features = torch.nn.Embedding.from_pretrained(self.n_feat_th, padding_idx=0, freeze=True)
        self.node_raw_features = torch.tensor(n_feat.astype(np.float32), requires_grad=False).to(device)
        self.edge_raw_features = torch.tensor(e_feat.astype(np.float32), requires_grad=False).to(device)
        
        self.feat_dim = self.n_feat_th.shape[1]
       
        self.n_feat_dim = self.feat_dim
        self.e_feat_dim = self.feat_dim
        self.model_dim = self.feat_dim
        self.time_dim = time_dim

        self.use_time = use_time
        # self.merge_layer = MergeLayer(self.feat_dim, self.feat_dim, self.feat_dim, self.feat_dim)
        

        if agg_method == 'attn':
            # self.logger.info('Aggregation uses attention model') ## or MultiHeadAttention ? 
            # self.attn_model_list = torch.nn.ModuleList([AttnModel(self.feat_dim, 
            #                                                    self.feat_dim, 
            #                                                    self.time_dim,
            #                                                    attn_mode=attn_mode, 
            #                                                    n_head=n_head, 
            #                                                    drop_out=drop_out) for _ in range(num_layers)])
            
            self.attn_model_list = nn.ModuleList([MultiHeadAttention(node_feat_dim=self.feat_dim,
                                                                      edge_feat_dim=self.feat_dim,
                                                                      time_feat_dim=self.time_dim,
                                                                      num_heads=n_head,
                                                                      dropout=drop_out) for _ in range(num_layers)])
                
                
        elif agg_method == 'lstm':
            # self.logger.info('Aggregation uses LSTM model')
            self.attn_model_list = torch.nn.ModuleList([LSTMPool(self.feat_dim,
                                                                 self.feat_dim,
                                                                 self.feat_dim) for _ in range(num_layers)])
        elif agg_method == 'mean':
            # self.logger.info('Aggregation uses constant mean model')
            self.attn_model_list = torch.nn.ModuleList([MeanPool(self.feat_dim,
                                                                 self.feat_dim) for _ in range(num_layers)])
        else:
        
            raise ValueError('invalid agg_method value, use attn or lstm')
        
        
        if use_time == 'time':
            # self.logger.info('Using time encoding')
            self.time_encoder = TimeEncode(self.time_dim)
           
        elif use_time == 'pos':
            assert(seq_len is not None)
            # self.logger.info('Using positional encoding')
            self.time_encoder = PosEncode(expand_dim=self.n_feat_th.shape[1], seq_len=seq_len)
        elif use_time == 'empty':
            # self.logger.info('Using empty encoding')
            self.time_encoder = EmptyEncode(expand_dim=self.n_feat_th.shape[1])
        else:
            raise ValueError('invalid time option!')
        
        self.affinity_score = MergeLayer(self.feat_dim, self.feat_dim, self.feat_dim, 1) #torch.nn.Bilinear(self.feat_dim, self.feat_dim, 1, bias=True)
        self.merge_layers = torch.nn.ModuleList([MergeLayer(dim1=self.feat_dim + self.time_dim,dim2=self.feat_dim, dim3=self.feat_dim,dim4=self.feat_dim) for _ in range(num_layers)])
        print('set',self.feat_dim + self.time_dim,self.feat_dim)

    def forward(self, src_idx_l, target_idx_l,edge_idxs, cut_time_l, num_neighbors=20):
        
        src_embed = self.tem_conv(src_idx_l, cut_time_l, self.num_layers, num_neighbors)
        target_embed = self.tem_conv(target_idx_l, cut_time_l, self.num_layers, num_neighbors)
        
 
        score = self.affinity_score(src_embed, target_embed).squeeze(dim=-1)
        

        return src_embed

    def contrast(self, src_idx_l, target_idx_l, background_idx_l, cut_time_l, num_neighbors=20):
        src_embed = self.tem_conv(src_idx_l, cut_time_l, self.num_layers, num_neighbors)
        target_embed = self.tem_conv(target_idx_l, cut_time_l, self.num_layers, num_neighbors)
        background_embed = self.tem_conv(background_idx_l, cut_time_l, self.num_layers, num_neighbors)
        pos_score = self.affinity_score(src_embed, target_embed).squeeze(dim=-1)
        neg_score = self.affinity_score(src_embed, background_embed).squeeze(dim=-1)
        return pos_score.sigmoid(), neg_score.sigmoid()

    def tem_conv(self, src_idx_l, cut_time_l, curr_layers, num_neighbors=20, candidate_weights_dict=None):
        assert(curr_layers >= 0)
        
        device = self.device
    
        batch_size = len(src_idx_l)
        
        src_node_batch_th = torch.from_numpy(src_idx_l).long().to(device)
        cut_time_l_th = torch.from_numpy(cut_time_l).float().to(device)
        
        cut_time_l_th = torch.unsqueeze(cut_time_l_th, dim=1)
        # query node always has the start time -> time span == 0
        src_node_t_embed = self.time_encoder(torch.zeros_like(cut_time_l_th))
        # src_node_feat = self.node_raw_features(src_node_batch_th)
        

        src_node_feat = self.node_raw_features[src_node_batch_th]

        

        if curr_layers == 0:
            return src_node_feat
        else:
            src_node_conv_feat = self.tem_conv(src_idx_l, 
                                           cut_time_l,
                                           curr_layers=curr_layers - 1, 
                                           num_neighbors=num_neighbors,
                                           candidate_weights_dict=candidate_weights_dict)
            
            
            src_ngh_node_batch, src_ngh_eidx_batch, src_ngh_t_batch = self.ngh_finder.get_temporal_neighbor( 
                                                                    src_idx_l, 
                                                                    cut_time_l, 
                                                                    num_neighbors)
            



            src_ngh_node_batch_th = torch.from_numpy(src_ngh_node_batch).long().to(device)
            src_ngh_eidx_batch = torch.from_numpy(src_ngh_eidx_batch).long().to(device)
            
            src_ngh_t_batch_delta = cut_time_l[:, np.newaxis] - src_ngh_t_batch

            src_ngh_t_batch_th = torch.from_numpy(src_ngh_t_batch_delta).float().to(device)
            
            # get previous layer's node features
            src_ngh_node_batch_flat = src_ngh_node_batch.flatten() #reshape(batch_size, -1)
            src_ngh_t_batch_flat = src_ngh_t_batch.flatten() #reshape(batch_size, -1)  
            src_ngh_node_conv_feat = self.tem_conv(src_ngh_node_batch_flat, 
                                                   src_ngh_t_batch_flat,
                                                   curr_layers=curr_layers - 1, 
                                                   num_neighbors=num_neighbors,
                                                   candidate_weights_dict=candidate_weights_dict)
            src_ngh_feat = src_ngh_node_conv_feat.view(batch_size, num_neighbors, -1)
            
            # get edge time features and node features
            # print('***************** src_ngh_t_batch_th', src_ngh_t_batch_th)
            src_ngh_t_embed = self.time_encoder(src_ngh_t_batch_th)
            # src_ngn_edge_feat = self.edge_raw_features(src_ngh_eidx_batch)
            src_ngn_edge_feat = self.edge_raw_features[src_ngh_eidx_batch]

            # attention aggregation
            mask = src_ngh_node_batch_th == 0
            attn_m = self.attn_model_list[curr_layers - 1]

            if candidate_weights_dict is not None:
                event_idxs = candidate_weights_dict['candidate_events']
                event_weights = candidate_weights_dict['edge_weights']

                ###### version 1, event_weights not [0, 1]
                position0 = src_ngh_node_batch_th == 0
                mask = torch.zeros_like(src_ngh_node_batch_th).to(dtype=torch.float32) # NOTE: for +, 0 mean no influence
                # import ipdb; ipdb.set_trace()
                for i, e_idx in enumerate(event_idxs):
                    indices = src_ngh_eidx_batch == e_idx
                    mask[indices] = event_weights[i]
                mask[position0] = -1e10 # addition attention, as 0 masks
     
            local, weight = attn_m(src_node_conv_feat, 
                                   src_node_t_embed,
                                   src_ngh_feat,
                                   src_ngh_t_embed, 
                                   src_ngn_edge_feat, 
                                   src_ngh_node_batch)
            

            ### merge layer ? merge atten and node feature 
            output = self.merge_layers[curr_layers - 1](x1=local, x2=src_node_feat) # ******* merge layer

            return output
    

    def get_embeddings(self, src_idx_l, target_idx_l, edge_idxs, cut_time_l, negative_nodes=None, num_neighbors=20, candidate_weights_dict=None):

        src_embed = self.tem_conv(src_idx_l, cut_time_l, self.num_layers, num_neighbors, candidate_weights_dict=candidate_weights_dict)
        target_embed = self.tem_conv(target_idx_l, cut_time_l, self.num_layers, num_neighbors, candidate_weights_dict=candidate_weights_dict)
        
     

        return src_embed, target_embed
    

    def set_neighbor_finder(self, neighbor_finder):
        self.ngh_finder = neighbor_finder
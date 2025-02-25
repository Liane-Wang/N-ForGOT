from random import random
from pandas import DataFrame
from pathlib import Path
from sklearn.utils import shuffle
from tqdm import tqdm
import numpy as np
import torch
import torch.nn as nn

from typing import Union, Optional, List
import itertools


def _set_tgat_data(all_events: DataFrame, target_event_idx: Union[int, List]):
    """ supporter for tgat """
    if isinstance(target_event_idx, (int, np.int64)):
        target_u = all_events.iloc[target_event_idx-1, 0]
        target_i = all_events.iloc[target_event_idx-1, 1]
        target_t = all_events.iloc[target_event_idx-1, 2]

        src_idx_l = np.array([target_u, ])
        target_idx_l = np.array([target_i, ])
        cut_time_l = np.array([target_t, ])
    elif isinstance(target_event_idx, list):
        # targets = all_events[all_events.e_idx.isin(target_event_idx)]
        targets = all_events.iloc[np.array(target_event_idx)-1] # faster?

        target_u = targets.u.values
        target_i = targets.i.values
        target_t = targets.ts.values

        src_idx_l = target_u
        target_idx_l = target_i
        cut_time_l = target_t
    else: 
        raise ValueError

    input_data = [src_idx_l, target_idx_l, cut_time_l]
    return input_data

def _create_explainer_input(model, model_name, all_events, candidate_events=None, event_idx=None, device=None):
    # DONE: explainer input should have both the target event and the event that we want to assign a weight to.

    if model_name in ['TGAT', 'TGN']:
        event_idx_u, event_idx_i, event_idx_t = _set_tgat_data(all_events, event_idx)
        # event_idx_new = _set_tgat_events_idxs(event_idx)
        event_idx_new = event_idx
        t_idx_u_emb = model.node_raw_features[ torch.tensor(event_idx_u, dtype=torch.int64, device=device), : ]
        t_idx_i_emb = model.node_raw_features[ torch.tensor(event_idx_i, dtype=torch.int64, device=device), : ]
        # import ipdb; ipdb.set_trace()
        t_idx_t_emb = model.time_encoder( torch.tensor(event_idx_t, dtype=torch.float32, device=device).reshape((1, -1)) ).reshape((1, -1))
        t_idx_e_emb = model.edge_raw_features[ torch.tensor([event_idx_new, ], dtype=torch.int64, device=device), : ]
        
        target_event_emb = torch.cat([t_idx_u_emb,t_idx_i_emb, t_idx_t_emb, t_idx_e_emb ], dim=1)
        
        candidate_events_u, candidate_events_i, candidate_events_t = _set_tgat_data(all_events, candidate_events)
        candidate_events_new = candidate_events

        candidate_u_emb = model.node_raw_features[ torch.tensor(candidate_events_u, dtype=torch.int64, device=device), : ]
        candidate_i_emb = model.node_raw_features[ torch.tensor(candidate_events_i, dtype=torch.int64, device=device), : ]
        candidate_t_emb = model.time_encoder( torch.tensor(candidate_events_t, dtype=torch.float32, device=device).reshape((1, -1)) ).reshape((len(candidate_events_t), -1))
        candidate_e_emb = model.edge_raw_features[ torch.tensor(candidate_events_new, dtype=torch.int64, device=device), : ]

        candiadte_events_emb = torch.cat([candidate_u_emb, candidate_i_emb, candidate_t_emb, candidate_e_emb], dim=1)

        input_expl = torch.cat([ target_event_emb.repeat(candiadte_events_emb.shape[0], 1),  candiadte_events_emb], dim=1)
        # import ipdb; ipdb.set_trace()
        return input_expl

    else:
        raise NotImplementedError


class BaseExplainerTG(object):
    def __init__(self, model, model_name: str, explainer_name: str, dataset_name: str, all_events: str, explanation_level: str, device, 
                verbose: bool = True, results_dir: Optional[str] = None, debug_mode: bool=True) -> None:
        """
        results_dir: dir for saving value results, e.g., fidelity_sparsity. Not mcts_node_list
        """
        self.model = model
        self.model_name = model_name
        self.explainer_name = explainer_name # self's name
        self.dataset_name = dataset_name
        self.all_events = all_events
        self.num_users = all_events.iloc[:, 0].max() + 1
        self.explanation_level = explanation_level
        
        self.device = device
        self.verbose = verbose
        self.results_dir = Path(results_dir)
        self.debug_mode = debug_mode
        
        self.model.eval()
        self.model.to(self.device)

        # construct TGNN reward function
        # self.tgnn_reward_wraper = TGNNRewardWraper(self.model, self.model_name, self.all_events, self.explanation_level)

    def find_candidates(self, target_event_idx):
        # TODO: implementation for other models
        # from tgnnexplainer.xgraph.dataset.utils_dataset import tgat_node_reindex
        # from tgnnexplainer.xgraph.method.tg_score import _set_tgat_events_idxs # NOTE: important

        if self.model_name in ['tgat', 'tgn']:
            # ngh_finder = self.model.ngh_finder
            # num_layers = self.model.num_layers
            # num_neighbors = self.model.num_neighbors # NOTE: important
            ngh_finder = self.model.base_model.ngh_finder
            num_layers = self.model.base_model.num_layers
            num_neighbors = self.model.base_model.num_neighbors # NOTE: important
            # edge_idx_preserve_list = self.ori_subgraph_df.e_idx.to_list() # NOTE: e_idx column

            u = self.all_events.iloc[target_event_idx-1, 0] # because target_event_idx should represent e_idx. e_idx = index + 1
            i = self.all_events.iloc[target_event_idx-1, 1]
            ts = self.all_events.iloc[target_event_idx-1, 2]

            # new_u, new_i = tgat_node_reindex(u, i, self.num_users)
            # accu_e_idx = [ [target_event_idx+1, target_event_idx+1]] # NOTE: for subsequent '-1' operation
            accu_e_idx = [ ] # NOTE: important?
            accu_node = [ [u, i,] ]
            accu_ts = [ [ts, ts,] ]
            
            for i in range(num_layers):
                last_nodes = accu_node[-1]
                last_ts = accu_ts[-1]
                # import ipdb; ipdb.set_trace()

                out_ngh_node_batch, out_ngh_eidx_batch, out_ngh_t_batch = ngh_finder.get_temporal_neighbor(
                                                                                    last_nodes, 
                                                                                    last_ts, 
                                                                                    num_neighbors=num_neighbors
                                                                                    )
                                                                                    # edge_idx_preserve_list=edge_idx_preserve_list, # NOTE: not needed?
                                                                                    # )
                
                out_ngh_node_batch = out_ngh_node_batch.flatten()
                out_ngh_eidx_batch = out_ngh_eidx_batch.flatten()
                out_ngh_t_batch = out_ngh_t_batch.flatten()
                
                mask = out_ngh_node_batch != 0
                out_ngh_node_batch = out_ngh_node_batch[mask]
                out_ngh_eidx_batch = out_ngh_eidx_batch[mask]
                out_ngh_t_batch = out_ngh_t_batch[mask]

                # import ipdb; ipdb.set_trace()

                out_ngh_node_batch = out_ngh_node_batch.tolist()
                out_ngh_t_batch = out_ngh_t_batch.tolist()
                out_ngh_eidx_batch = (out_ngh_eidx_batch).tolist() 

                accu_node.append(out_ngh_node_batch)
                accu_ts.append(out_ngh_t_batch)
                accu_e_idx.append(out_ngh_eidx_batch)
                # import ipdb; ipdb.set_trace()

            unique_e_idx = np.array(list(itertools.chain.from_iterable(accu_e_idx)))
            unique_e_idx = unique_e_idx[ unique_e_idx != 0 ] # NOTE: 0 are padded e_idxs
            # unique_e_idx = unique_e_idx - 1 # NOTE: -1, because ngh_finder stored +1 e_idxs
            unique_e_idx = np.unique(unique_e_idx).tolist()

            # TODO: to test self.base_events = unique_e_idx, will this influence the speed?

            
        else:
            raise NotImplementedError
        
        candidate_events = unique_e_idx
        threshold_num = 20
        if len(candidate_events) > threshold_num:
            candidate_events = candidate_events[-threshold_num:]
            candidate_events = sorted(candidate_events)
        # import ipdb; ipdb.set_trace()
        
        if self.debug_mode:
            print(f'{len(unique_e_idx)} seen events, used {len(candidate_events)} as candidates:')
            print(candidate_events)
        
        return candidate_events, unique_e_idx
    
    # def _set_ori_subgraph(self, num_hops, event_idx):
    #     subgraph_df = k_hop_temporal_subgraph(self.all_events, num_hops=num_hops, event_idx=event_idx)
    #     self.ori_subgraph_df = subgraph_df


    def _set_candidate_events(self, event_idx):
        self.candidate_events, unique_e_idx = self.find_candidates(event_idx)
        # self.candidate_events = shuffle( candidate_events ) # strategy 1
        # self.candidate_events = candidate_events # strategy 2
        # self.candidate_events.reverse()
        # self.candidate_events = candidate_events # strategy 3
        candidate_events_set_ = set(self.candidate_events)
        assert hasattr(self, 'ori_subgraph_df')
        # self.base_events = list(filter(lambda x: x not in candidate_events_set_, self.ori_subgraph_df.e_idx.values) ) # NOTE: ori_subgraph_df.e_idx.values
        self.base_events = list(filter(lambda x: x not in candidate_events_set_, unique_e_idx) ) # NOTE: an importanct change, need test. largely influence the speed!



    # def _set_tgnn_wraper(self, event_idx):
    #     assert hasattr(self, 'ori_subgraph_df')
    #     self.tgnn_reward_wraper.compute_original_score(self.base_events+self.candidate_events, event_idx)
    
    def _initialize(self, event_idx):
        # self._set_ori_subgraph(num_hops=3, event_idx=event_idx)
        self._set_candidate_events(event_idx)
        # self._set_tgnn_wraper(event_idx)
        # self.candidate_initial_weights = None
        np.random.seed(1)
        self.candidate_initial_weights = { e_idx: np.random.random() for e_idx in self.candidate_events }

    @staticmethod
    def _score_path(results_dir, model_name, dataset_name, explainer_name, event_idx,):
        """
        only for baseline explainer, save their computed candidate scores.
        """
        score_filename = results_dir/f'{model_name}_{dataset_name}_{explainer_name}_{event_idx}_candidate_scores.csv'
        return score_filename

    def _save_candidate_scores(self, candidate_weights, event_idx):
        """
        only for baseline explainer, save their computed candidate scores.
        """
        assert isinstance(candidate_weights, dict)
        filename = self._score_path(self.results_dir, self.model_name, self.dataset_name, self.explainer_name, event_idx)
        data_dict = {
            'candidates': [],
            'scores': []
        }
        for k, v in candidate_weights.items():
            data_dict['candidates'].append(k)
            data_dict['scores'].append(v)
        
        df = DataFrame(data_dict)
        df.to_csv(filename, index=False)
        print(f'candidate scores saved at {filename}')



class PGExplainerExt(BaseExplainerTG):
    def __init__(self, model, model_name: str, explainer_name: str, dataset_name: str, input_dim: int, args,
                 all_events: DataFrame,  explanation_level: str, device, verbose: bool = True, results_dir = None, debug_mode=True,
                 # specific params for PGExplainerExt
                 train_epochs: int = 50, explainer_ckpt_dir = None, reg_coefs = None, batch_size = 64, lr=1e-4
                ):
        super(PGExplainerExt, self).__init__(model=model,
                                              model_name=model_name,
                                              explainer_name=explainer_name,
                                              dataset_name=dataset_name,
                                              all_events=all_events,
                                              explanation_level=explanation_level,
                                              device=device,
                                              verbose=verbose,
                                              results_dir=results_dir,
                                              debug_mode=debug_mode
                                              )
        self.train_epochs = train_epochs
        self.explainer_ckpt_dir = explainer_ckpt_dir
        self.reg_coefs = reg_coefs
        self.batch_size = batch_size
        self.lr = lr
        self.expl_input_dim = None
        self._init_explainer()
        self.args = args
        
    @staticmethod
    def _create_explainer(model, model_name, device):

        expl_input_dim = model.feat_dim * 8

        # if model_name == 'tgat':
        #     expl_input_dim = model.model_dim * 8 # 2 * (dim_u + dim_i + dim_t + dim_e)
        # elif model_name == 'tgn':
        #     expl_input_dim = model.n_node_features * 8
        # else:
        #     raise NotImplementedError

        explainer_model = nn.Sequential(
            nn.Linear(expl_input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 1),  ##### version 1
            # nn.Sigmoid(), ##### version 2
        )
        explainer_model = explainer_model.to(device)
        return explainer_model

    # @staticmethod
    def _ckpt_path(self, ckpt_dir, model_name, dataset_name, explainer_name, epoch=None):
        if epoch is None:
            # return Path(ckpt_dir)/f'{model_name}_{dataset_name}_{explainer_name}_expl_ckpt.pt'
            return f'./checkpoints/{self.args.model}/' + str(vars(self.args)) + '_expl_ckpt.pth'
        else:
            # return Path(ckpt_dir)/f'{model_name}_{dataset_name}_{explainer_name}_expl_ckpt_ep{epoch}.pt'
            return f'./checkpoints/{self.args.model}/' + str(vars(self.args)) + f'_expl_ckpt.pth{epoch}'
    
    def _init_explainer(self):
        self.explainer_model = self._create_explainer(self.model.base_model, self.model_name, self.device)

    def __call__(self, node_idxs: Union[int, None] = None, event_idxs: Union[int, None] = None):
        self.explainer_ckpt_path = self._ckpt_path(self.explainer_ckpt_dir, self.model_name, self.dataset_name, self.explainer_name)
        self.explain_event_idxs = event_idxs
        
        if not self.explainer_ckpt_path.exists():
            self._train() # we need to train the explainer first
        else:
            state_dict = torch.load(self.explainer_ckpt_path)
            self.explainer_model.load_state_dict(state_dict)

        results_list = []
        for i, event_idx in enumerate(event_idxs):
            print(f'\nexplain {i}-th: {event_idx}')
            self._initialize(event_idx)
            candidate_weights = self.explain(event_idx=event_idx)
            results_list.append( [ list(candidate_weights.keys()), list(candidate_weights.values()) ] )

            # self._save_candidate_scores(candidate_weights, event_idx)

        # import ipdb; ipdb.set_trace()
        return results_list

    ## TODO: This function is of great importance to modify!!!!!!!!!!!!    
    def _tg_predict(self, event_idx, use_explainer=False):
        if self.model_name in ['tgat', 'tgn']:
            src_idx_l, target_idx_l, cut_time_l = _set_tgat_data(self.all_events, event_idx)
            edge_weights = None
            if use_explainer:
                # candidate_events_new = _set_tgat_events_idxs(self.candidate_events) # these temporal edges to alter attn weights in tgat
                input_expl = _create_explainer_input(self.model.base_model, self.model_name, self.all_events, \
                    candidate_events=self.candidate_events, event_idx=event_idx, device=self.device)
                # import ipdb; ipdb.set_trace()
                edge_weights = self.explainer_model(input_expl)
                candidate_weights_dict = {'candidate_events': torch.tensor(self.candidate_events, dtype=torch.int64, device=self.device),
                                    'edge_weights': edge_weights,
                }
            else:
                candidate_weights_dict = None
            # NOTE: use the 'src_ngh_eidx_batch' in module to locate mask fill positions
            output = self.model(src_idx_l, target_idx_l, cut_time_l, logit=True, candidate_weights_dict=candidate_weights_dict)
            return output, edge_weights

        else: 
            raise NotImplementedError

    def _loss(self, masked_pred, original_pred, mask, reg_coefs):
        # TODO: improve the loss?
        size_reg = reg_coefs[0]
        entropy_reg = reg_coefs[1]

        # Regularization losses
        size_loss = torch.sum(mask) * size_reg
        # mask_ent_reg = -mask * torch.log(mask) - (1 - mask) * torch.log(1 - mask)
        # mask_ent_loss = entropy_reg * torch.mean(mask_ent_reg)

        # Explanation loss
        if original_pred > 0: # larger better
            error_loss = masked_pred - original_pred
        else:
            error_loss = original_pred - masked_pred
        error_loss = error_loss * -1 # to minimize 
        
        # cce_loss = torch.nn.functional.mse_loss(masked_pred, original_pred)

        #return cce_loss
        # return cce_loss + size_loss + mask_ent_loss
        # return cce_loss + size_loss
        # import ipdb; ipdb.set_trace()

        return error_loss
    
    def _obtain_train_idxs(self,):
        size = min(1000, int(len(self.all_events)*0.4))
        # np.random.seed( np.random.randint(10000) )
        # if self.dataset_name in ['wikipedia', 'reddit']:
        train_e_idxs = np.random.randint(int(len(self.all_events)*0.2), int(len(self.all_events)*0.6), (size, ))
        train_e_idxs = shuffle(train_e_idxs) # TODO: not shuffle?
        # elif self.dataset_name in ['simulate_v1', 'simulate_v2']:
        #     positive_indices = self.all_events.label == 1 
        #     pos_events = self.all_events[positive_indices].e_idx.values
        #     train_e_idxs = np.random.choice(pos_events, size=size, replace=False)

        return train_e_idxs



    def _train(self,):
        self.explainer_model.train()
        optimizer = torch.optim.Adam(self.explainer_model.parameters(), lr=self.lr)

        for e in range(self.train_epochs):
            train_e_idxs = self._obtain_train_idxs()

            optimizer.zero_grad()
            loss = torch.tensor([0], dtype=torch.float32, device=self.device)
            loss_list = []
            counter = 0
            skipped_num = 0

            for i, event_idx in tqdm(enumerate(train_e_idxs), total=len(train_e_idxs), desc=f'epoch {e}' ): # training
                self._initialize(event_idx) # NOTE: needed
                if len(self.candidate_events) == 0: # skip bad samples
                    skipped_num += 1
                    continue

                original_pred, mask_values_ = self._tg_predict(event_idx, use_explainer=False)
                masked_pred, mask_values = self._tg_predict(event_idx, use_explainer=True)

                id_loss = self._loss(masked_pred, original_pred, mask_values, self.reg_coefs)
                # import ipdb; ipdb.set_trace()
                id_loss = id_loss.flatten()
                assert len(id_loss) == 1

                loss += id_loss
                loss_list.append(id_loss.cpu().detach().item())
                counter += 1

                if counter % self.batch_size == 0:
                    loss = loss/self.batch_size
                    loss.backward()
                    optimizer.step()
                    loss = torch.tensor([0], dtype=torch.float32, device=self.device)
                    optimizer.zero_grad()
        
            # import ipdb; ipdb.set_trace()
            state_dict = self.explainer_model.state_dict()
            ckpt_save_path = self._ckpt_path(self.explainer_ckpt_dir, self.model_name, self.dataset_name, self.explainer_name, epoch=e)
            torch.save(state_dict, ckpt_save_path)
            tqdm.write(f"epoch {e} loss epoch {np.mean(loss_list)}, skipped: {skipped_num}, ckpt saved: {ckpt_save_path}")

        state_dict = self.explainer_model.state_dict()
        torch.save(state_dict, self.explainer_ckpt_path)
        print('train finished')
        print(f'explainer ckpt saved at {str(self.explainer_ckpt_path)}')

    def explain(self, node_idx=None, event_idx=None):
        self.explainer_model.eval()
        input_expl = _create_explainer_input(self.model.base_model, self.model_name, self.all_events, \
            candidate_events=self.candidate_events, event_idx=event_idx, device=self.device)
        event_idx_scores = self.explainer_model(input_expl) # compute importance scores
        event_idx_scores = event_idx_scores.cpu().detach().numpy().flatten()

        # the same as Attn explainer
        candidate_weights = { self.candidate_events[i]: event_idx_scores[i] for i in range(len(self.candidate_events)) }
        candidate_weights = dict( sorted(candidate_weights.items(), key=lambda x: x[1], reverse=True) ) # NOTE: descending, important

        return candidate_weights
    
    @staticmethod
    def expose_explainer_model(model, model_name, explainer_name, dataset_name, ckpt_dir, device):
        explainer_model = PGExplainerExt._create_explainer(model, model_name, device)
        explainer_ckpt_path = PGExplainerExt._ckpt_path(ckpt_dir, model_name, dataset_name, explainer_name)

        state_dict = torch.load(explainer_ckpt_path)
        explainer_model.load_state_dict(state_dict)

        return explainer_model, explainer_ckpt_path

import numpy as np
import torch
from torch import device, nn
import torch.nn.functional as F
from utils.log_and_checkpoints import get_checkpoint_path
import dill 
from copy import deepcopy
import os
from collections import OrderedDict
import random
from utils.data_processing import Data

class EarlyStopMonitor(object):
    def __init__(self, max_round, higher_better=True, tolerance=1e-10):
        self.max_round = max_round
        self.num_round = 0

        self.epoch_count = 0
        self.best_epoch = 0

        self.last_best = None
        self.higher_better = higher_better
        self.tolerance = tolerance

    def early_stop_check(
        self,
        curr_val,
    ):
        if not self.higher_better:
            curr_val *= -1
        if self.last_best is None:
            self.last_best = curr_val
            self.num_round = 0
            self.best_epoch = self.epoch_count
            
        elif (curr_val - self.last_best) / np.abs(self.last_best) > self.tolerance:
            self.last_best = curr_val
            self.num_round = 0
            self.best_epoch = self.epoch_count
        else:
            self.num_round += 1

        self.epoch_count += 1

        return self.num_round >= self.max_round
    


class RandEdgeSampler(object):
    def __init__(self, src_list, dst_list, seed=None):
        self.seed = None
        self.src_list = np.unique(src_list)
        self.dst_list = np.unique(dst_list)
        self.len_src = len(self.src_list)
        self.len_dst = len(self.dst_list)
        if seed is not None:
            self.seed = seed
            self.random_state = np.random.RandomState(self.seed)

    def sample(self, size):
        if self.seed is None:
            src_idx = np.random.randint(0, self.len_src, size)
            dst_idx = np.random.randint(0, self.len_dst, size)
        else:
            src_idx = self.random_state.randint(0, self.len_src, size)
            dst_idx = self.random_state.randint(0, self.len_dst, size)
        return self.src_list[src_idx], self.dst_list[dst_idx]

    def reset_random_state(self):
        self.random_state = np.random.RandomState(self.seed)


class NeighborFinder:

    def __init__(self, adj_list: list, sample_neighbor_strategy: str = 'uniform', time_scaling_factor: float = 0.0, seed: int = None):
        """
        Neighbor sampler.
        :param adj_list: list, list of list, where each element is a list of triple tuple (node_id, edge_id, timestamp)
        :param sample_neighbor_strategy: str, how to sample historical neighbors, 'uniform', 'recent', or 'time_interval_aware'
        :param time_scaling_factor: float, a hyper-parameter that controls the sampling preference with time interval,
        a large time_scaling_factor tends to sample more on recent links, this parameter works when sample_neighbor_strategy == 'time_interval_aware'
        :param seed: int, random seed
        """
        self.sample_neighbor_strategy = sample_neighbor_strategy
        self.seed = seed

        # list of each node's neighbor ids, edge ids and interaction times, which are sorted by interaction times
        self.nodes_neighbor_ids = []
        self.nodes_edge_ids = []
        self.nodes_neighbor_times = []

        if self.sample_neighbor_strategy == 'time_interval_aware':
            self.nodes_neighbor_sampled_probabilities = []
            self.time_scaling_factor = time_scaling_factor

        # the list at the first position in adj_list is empty, hence, sorted() will return an empty list for the first position
        # its corresponding value in self.nodes_neighbor_ids, self.nodes_edge_ids, self.nodes_neighbor_times will also be empty with length 0
        for node_idx, per_node_neighbors in enumerate(adj_list):
            # per_node_neighbors is a list of tuples (neighbor_id, edge_id, timestamp)
            # sort the list based on timestamps, sorted() function is stable
            # Note that sort the list based on edge id is also correct, as the original data file ensures the interactions are chronological
            sorted_per_node_neighbors = sorted(per_node_neighbors, key=lambda x: x[2])
            self.nodes_neighbor_ids.append(np.array([x[0] for x in sorted_per_node_neighbors]))
            self.nodes_edge_ids.append(np.array([x[1] for x in sorted_per_node_neighbors]))
            self.nodes_neighbor_times.append(np.array([x[2] for x in sorted_per_node_neighbors]))

            # additional for time interval aware sampling strategy (proposed in CAWN paper)
            if self.sample_neighbor_strategy == 'time_interval_aware':
                self.nodes_neighbor_sampled_probabilities.append(self.compute_sampled_probabilities(np.array([x[2] for x in sorted_per_node_neighbors])))

        if self.seed is not None:
            self.random_state = np.random.RandomState(self.seed)

    def compute_sampled_probabilities(self, node_neighbor_times: np.ndarray):
        """
        compute the sampled probabilities of historical neighbors based on their interaction times
        :param node_neighbor_times: ndarray, shape (num_historical_neighbors, )
        :return:
        """
        if len(node_neighbor_times) == 0:
            return np.array([])
        # compute the time delta with regard to the last time in node_neighbor_times
        node_neighbor_times = node_neighbor_times - np.max(node_neighbor_times)
        # compute the normalized sampled probabilities of historical neighbors
        exp_node_neighbor_times = np.exp(self.time_scaling_factor * node_neighbor_times)
        sampled_probabilities = exp_node_neighbor_times / np.cumsum(exp_node_neighbor_times)
        # note that the first few values in exp_node_neighbor_times may be all zero, which make the corresponding values in sampled_probabilities
        # become nan (divided by zero), so we replace the nan by a very large negative number -1e10 to denote the sampled probabilities
        sampled_probabilities[np.isnan(sampled_probabilities)] = -1e10
        return sampled_probabilities

    def find_neighbors_before(self, node_id: int, interact_time: float, return_sampled_probabilities: bool = False):
        """
        extracts all the interactions happening before interact_time (less than interact_time) for node_id in the overall interaction graph
        the returned interactions are sorted by time.
        :param node_id: int, node id
        :param interact_time: float, interaction time
        :param return_sampled_probabilities: boolean, whether return the sampled probabilities of neighbors
        :return: neighbors, edge_ids, timestamps and sampled_probabilities (if return_sampled_probabilities is True) with shape (historical_nodes_num, )
        """
        # return index i, which satisfies list[i - 1] < v <= list[i]
        # return 0 for the first position in self.nodes_neighbor_times since the value at the first position is empty
        if node_id >= len(self.nodes_neighbor_times):
            print(node_id)
            print(len(self.nodes_neighbor_times))
        i = np.searchsorted(self.nodes_neighbor_times[node_id], interact_time)

        if return_sampled_probabilities:
            return self.nodes_neighbor_ids[node_id][:i], self.nodes_edge_ids[node_id][:i], self.nodes_neighbor_times[node_id][:i], \
                   self.nodes_neighbor_sampled_probabilities[node_id][:i]
        else:
            return self.nodes_neighbor_ids[node_id][:i], self.nodes_edge_ids[node_id][:i], self.nodes_neighbor_times[node_id][:i], None

    def get_historical_neighbors(self, node_ids: np.ndarray, node_interact_times: np.ndarray, num_neighbors: int = 20):
        """
        get historical neighbors of nodes in node_ids with interactions before the corresponding time in node_interact_times
        :param node_ids: ndarray, shape (batch_size, ) or (*, ), node ids
        :param node_interact_times: ndarray, shape (batch_size, ) or (*, ), node interaction times
        :param num_neighbors: int, number of neighbors to sample for each node
        :return:
        """
        assert num_neighbors > 0, 'Number of sampled neighbors for each node should be greater than 0!'
        # All interactions described in the following three matrices are sorted in each row by time
        # each entry in position (i,j) represents the id of the j-th dst node of src node node_ids[i] with an interaction before node_interact_times[i]
        # ndarray, shape (batch_size, num_neighbors)
        nodes_neighbor_ids = np.zeros((len(node_ids), num_neighbors)).astype(np.longlong)
        # each entry in position (i,j) represents the id of the edge with src node node_ids[i] and dst node nodes_neighbor_ids[i][j] with an interaction before node_interact_times[i]
        # ndarray, shape (batch_size, num_neighbors)
        nodes_edge_ids = np.zeros((len(node_ids), num_neighbors)).astype(np.longlong)
        # each entry in position (i,j) represents the interaction time between src node node_ids[i] and dst node nodes_neighbor_ids[i][j], before node_interact_times[i]
        # ndarray, shape (batch_size, num_neighbors)
        nodes_neighbor_times = np.zeros((len(node_ids), num_neighbors)).astype(np.float32)

        # extracts all neighbors ids, edge ids and interaction times of nodes in node_ids, which happened before the corresponding time in node_interact_times
        for idx, (node_id, node_interact_time) in enumerate(zip(node_ids, node_interact_times)):
            # find neighbors that interacted with node_id before time node_interact_time
            node_neighbor_ids, node_edge_ids, node_neighbor_times, node_neighbor_sampled_probabilities = \
                self.find_neighbors_before(node_id=node_id, interact_time=node_interact_time, return_sampled_probabilities=self.sample_neighbor_strategy == 'time_interval_aware')

            if len(node_neighbor_ids) > 0:
                if self.sample_neighbor_strategy in ['uniform', 'time_interval_aware']:
                    # when self.sample_neighbor_strategy == 'uniform', we shuffle the data before sampling with node_neighbor_sampled_probabilities as None
                    # when self.sample_neighbor_strategy == 'time_interval_aware', we sample neighbors based on node_neighbor_sampled_probabilities
                    # for time_interval_aware sampling strategy, we additionally use softmax to make the sum of sampled probabilities be 1
                    if node_neighbor_sampled_probabilities is not None:
                        # for extreme case that node_neighbor_sampled_probabilities only contains -1e10, which will make the denominator of softmax be zero,
                        # torch.softmax() function can tackle this case
                        node_neighbor_sampled_probabilities = torch.softmax(torch.from_numpy(node_neighbor_sampled_probabilities).float(), dim=0).numpy()
                    if self.seed is None:
                        sampled_indices = np.random.choice(a=len(node_neighbor_ids), size=num_neighbors, p=node_neighbor_sampled_probabilities)
                    else:
                        sampled_indices = self.random_state.choice(a=len(node_neighbor_ids), size=num_neighbors, p=node_neighbor_sampled_probabilities)

                    nodes_neighbor_ids[idx, :] = node_neighbor_ids[sampled_indices]
                    nodes_edge_ids[idx, :] = node_edge_ids[sampled_indices]
                    nodes_neighbor_times[idx, :] = node_neighbor_times[sampled_indices]

                    # resort based on timestamps, return the ids in sorted increasing order, note this maybe unstable when multiple edges happen at the same time
                    # (we still do this though this is unnecessary for TGAT or CAWN to guarantee the order of nodes,
                    # since TGAT computes in an order-agnostic manner with relative time encoding, and CAWN computes for each walk while the sampled nodes are in different walks)
                    sorted_position = nodes_neighbor_times[idx, :].argsort()
                    nodes_neighbor_ids[idx, :] = nodes_neighbor_ids[idx, :][sorted_position]
                    nodes_edge_ids[idx, :] = nodes_edge_ids[idx, :][sorted_position]
                    nodes_neighbor_times[idx, :] = nodes_neighbor_times[idx, :][sorted_position]
                elif self.sample_neighbor_strategy == 'recent':
                    # Take most recent interactions with number num_neighbors
                    node_neighbor_ids = node_neighbor_ids[-num_neighbors:]
                    node_edge_ids = node_edge_ids[-num_neighbors:]
                    node_neighbor_times = node_neighbor_times[-num_neighbors:]

                    # put the neighbors' information at the back positions
                    nodes_neighbor_ids[idx, num_neighbors - len(node_neighbor_ids):] = node_neighbor_ids
                    nodes_edge_ids[idx, num_neighbors - len(node_edge_ids):] = node_edge_ids
                    nodes_neighbor_times[idx, num_neighbors - len(node_neighbor_times):] = node_neighbor_times
                else:
                    raise ValueError(f'Not implemented error for sample_neighbor_strategy {self.sample_neighbor_strategy}!')

        # three ndarrays, with shape (batch_size, num_neighbors)
        return nodes_neighbor_ids, nodes_edge_ids, nodes_neighbor_times

    def get_multi_hop_neighbors(self, num_hops: int, node_ids: np.ndarray, node_interact_times: np.ndarray, num_neighbors: int = 20):
        """
        get historical neighbors of nodes in node_ids within num_hops hops
        :param num_hops: int, number of sampled hops
        :param node_ids: ndarray, shape (batch_size, ), node ids
        :param node_interact_times: ndarray, shape (batch_size, ), node interaction times
        :param num_neighbors: int, number of neighbors to sample for each node
        :return:
        """
        assert num_hops > 0, 'Number of sampled hops should be greater than 0!'

        # get the temporal neighbors at the first hop
        # nodes_neighbor_ids, nodes_edge_ids, nodes_neighbor_times -> ndarray, shape (batch_size, num_neighbors)
        nodes_neighbor_ids, nodes_edge_ids, nodes_neighbor_times = self.get_historical_neighbors(node_ids=node_ids,
                                                                                                 node_interact_times=node_interact_times,
                                                                                                 num_neighbors=num_neighbors)
        # three lists to store the neighbor ids, edge ids and interaction timestamp information
        nodes_neighbor_ids_list = [nodes_neighbor_ids]
        nodes_edge_ids_list = [nodes_edge_ids]
        nodes_neighbor_times_list = [nodes_neighbor_times]
        for hop in range(1, num_hops):
            # get information of neighbors sampled at the current hop
            # three ndarrays, with shape (batch_size * num_neighbors ** hop, num_neighbors)
            nodes_neighbor_ids, nodes_edge_ids, nodes_neighbor_times = self.get_historical_neighbors(node_ids=nodes_neighbor_ids_list[-1].flatten(),
                                                                                                     node_interact_times=nodes_neighbor_times_list[-1].flatten(),
                                                                                                     num_neighbors=num_neighbors)
            # three ndarrays with shape (batch_size, num_neighbors ** (hop + 1))
            nodes_neighbor_ids = nodes_neighbor_ids.reshape(len(node_ids), -1)
            nodes_edge_ids = nodes_edge_ids.reshape(len(node_ids), -1)
            nodes_neighbor_times = nodes_neighbor_times.reshape(len(node_ids), -1)

            nodes_neighbor_ids_list.append(nodes_neighbor_ids)
            nodes_edge_ids_list.append(nodes_edge_ids)
            nodes_neighbor_times_list.append(nodes_neighbor_times)

        # tuple, each element in the tuple is a list of num_hops ndarrays, each with shape (batch_size, num_neighbors ** current_hop)
        return nodes_neighbor_ids_list, nodes_edge_ids_list, nodes_neighbor_times_list

    def get_all_first_hop_neighbors(self, node_ids: np.ndarray, node_interact_times: np.ndarray):
        """
        get historical neighbors of nodes in node_ids at the first hop with max_num_neighbors as the maximal number of neighbors (make the computation feasible)
        :param node_ids: ndarray, shape (batch_size, ), node ids
        :param node_interact_times: ndarray, shape (batch_size, ), node interaction times
        :return:
        """
        # three lists to store the first-hop neighbor ids, edge ids and interaction timestamp information, with batch_size as the list length
        nodes_neighbor_ids_list, nodes_edge_ids_list, nodes_neighbor_times_list = [], [], []
        # get the temporal neighbors at the first hop
        for idx, (node_id, node_interact_time) in enumerate(zip(node_ids, node_interact_times)):
            # find neighbors that interacted with node_id before time node_interact_time
            node_neighbor_ids, node_edge_ids, node_neighbor_times, _ = self.find_neighbors_before(node_id=node_id,
                                                                                                  interact_time=node_interact_time,
                                                                                                  return_sampled_probabilities=False)
            nodes_neighbor_ids_list.append(node_neighbor_ids)
            nodes_edge_ids_list.append(node_edge_ids)
            nodes_neighbor_times_list.append(node_neighbor_times)

        return nodes_neighbor_ids_list, nodes_edge_ids_list, nodes_neighbor_times_list

    def reset_random_state(self):
        """
        reset the random state by self.seed
        :return:
        """
        self.random_state = np.random.RandomState(self.seed)


def get_neighbor_sampler(data: Data, sample_neighbor_strategy: str = 'uniform', time_scaling_factor: float = 0.0, seed: int = None):
    """
    get neighbor sampler
    :param data: Data
    :param sample_neighbor_strategy: str, how to sample historical neighbors, 'uniform', 'recent', or 'time_interval_aware''
    :param time_scaling_factor: float, a hyper-parameter that controls the sampling preference with time interval,
    a large time_scaling_factor tends to sample more on recent links, this parameter works when sample_neighbor_strategy == 'time_interval_aware'
    :param seed: int, random seed
    :return:
    """
    max_node_id = max(data.src.max(), data.dst.max())
    max_node_id = 1500000
    # the adjacency vector stores edges for each node (source or destination), undirected
    # adj_list, list of list, where each element is a list of triple tuple (node_id, edge_id, timestamp)
    # the list at the first position in adj_list is empty
    adj_list = [[] for _ in range(max_node_id + 1)]
    for src_node_id, dst_node_id, edge_id, node_interact_time in zip(data.src, data.dst, data.edge_idxs, data.timestamps):
        adj_list[src_node_id].append((dst_node_id, edge_id, node_interact_time))
        adj_list[dst_node_id].append((src_node_id, edge_id, node_interact_time))

    return NeighborFinder(adj_list=adj_list, sample_neighbor_strategy=sample_neighbor_strategy, time_scaling_factor=time_scaling_factor, seed=seed)



class NeighborFinder_old:
    def __init__(self, adj_list, uniform=False, seed=None):
        self.node_to_neighbors = []
        self.node_to_edge_idxs = []
        self.node_to_edge_time_stamps = []
        for neighbors in adj_list:
            # format of neighbors : (neighbor, edge_idx, timestamp)
            # sorted base on timestamp
            sorted_neighbor = sorted(neighbors, key=lambda x: x[2])
            self.node_to_neighbors.append(np.array([x[0] for x in sorted_neighbor]))
            self.node_to_edge_idxs.append(np.array([x[1] for x in sorted_neighbor]))
            self.node_to_edge_time_stamps.append(
                np.array([x[2] for x in sorted_neighbor])
            )
        # [ngh 1-1,ngh 1-2,...ngh 1-n, ngh 2-1,...]
        self.uniform = uniform
        if self.uniform:
            self.sample_neighbor_strategy = 'uniform'
        else:
            self.sample_neighbor_strategy = 'recent'
        if seed is not None:
            self.seed = seed
            self.random_state = np.random.RandomState(self.seed)

    def find_before(self, src_idx, cut_time):
        """
        Extracts all the interactions happening before cut_time
        for user src_idx in the overall interaction graph.
        The returned interactions are sorted by time.

        Returns 3 lists: neighbors, edge_idxs, timestamps

        """
        if src_idx >= len(self.node_to_edge_time_stamps):
            print(src_idx)
            print(len(self.node_to_edge_time_stamps))
        i = np.searchsorted(self.node_to_edge_time_stamps[src_idx], cut_time)
        return (
            self.node_to_neighbors[src_idx][:i],
            self.node_to_edge_idxs[src_idx][:i],
            self.node_to_edge_time_stamps[src_idx][:i],
        )

    def get_temporal_neighbor(self, src_nodes, timestamps, n_neighbors=20, batch_wise_edge_time=True):
        """
        Given a list of users ids and relative cut times, extracts
        a sampled temporal neighborhood of each user in the list.

        Params
        ------
        src_idx_l: List[int]
        cut_time_l: List[float],
        num_neighbors: int
        """

        assert (len(src_nodes)) == len(timestamps)

        tmp_n_neighbors = n_neighbors if n_neighbors > 0 else 1
        # All interactions described in these matrices are sorted in each row by time
        neighbors = np.zeros((len(src_nodes), tmp_n_neighbors)).astype(np.float32)
        # each entry in position (i,j) represent the id of the item
        # targeted by user src_idx_l[i] with an interaction happening
        # before cut_time_l[i]

        edge_times = np.zeros((len(src_nodes), tmp_n_neighbors)).astype(np.float32)
        # edge_times[:] = 1e15
        # each entry in position (i,j) represent the timestamp of
        # an interaction between user src_idx_l[i] and item neighbors[i,j]
        # happening before cut_time_l[i]

        edge_idxs = np.zeros((len(src_nodes), tmp_n_neighbors)).astype(np.int32)
        # each entry in position (i,j) represent the interaction index
        # of an interaction between user src_idx_l[i] and item neighbors[i,j]
        # happening before cut_time_l[i]

        for i, (src_node, timestamp) in enumerate(zip(src_nodes, timestamps)):
            src_neighbors, src_edge_idxs, src_edge_times = self.find_before(
                src_node, timestamp
            )
            # extracts all neighbors, interactions indexes and timestamps
            # of all interactions of user source_node happening before cut_time
            if len(src_neighbors) > 0 and n_neighbors > 0:
                if self.uniform:
                    # if we are applying uniform sampling, shuffles the data
                    # above before sampling
                    sample_idx = np.random.randint(0, len(src_neighbors), n_neighbors)

                    neighbors[i, :] = src_neighbors[sample_idx]
                    edge_times[i, :] = src_edge_times[sample_idx]
                    edge_idxs[i, :] = src_edge_idxs[sample_idx]

                    # re-sort based on time
                    pos = edge_times[i, :].argsort()
                    neighbors[i, :] = neighbors[i, :][pos]
                    edge_times[i, :] = edge_times[i, :][pos]
                    edge_idxs[i, :] = edge_idxs[i, :][pos]
                else:
                    # Take most recent interactions
                    source_edge_times = src_edge_times[-n_neighbors:]
                    source_neighbors = src_neighbors[-n_neighbors:]
                    source_edge_idxs = src_edge_idxs[-n_neighbors:]

                    assert len(source_neighbors) <= n_neighbors
                    assert len(source_edge_times) <= n_neighbors
                    assert len(source_edge_idxs) <= n_neighbors

                    neighbors[
                        i, n_neighbors - len(source_neighbors) :
                    ] = source_neighbors
                    edge_times[
                        i, n_neighbors - len(source_edge_times) :
                    ] = source_edge_times
                    edge_idxs[
                        i, n_neighbors - len(source_edge_idxs) :
                    ] = source_edge_idxs
        
                # if batch_wise_edge_time:
                #     min_time = min(timestamps)
                #     edge_times[i, n_neighbors - len(source_edge_times) :] = source_edge_times - min_time

        return neighbors, edge_idxs, edge_times
    
    def get_all_first_hop_neighbors(self, node_ids: np.ndarray, node_interact_times: np.ndarray):
        """
        get historical neighbors of nodes in node_ids at the first hop with max_num_neighbors as the maximal number of neighbors (make the computation feasible)
        :param node_ids: ndarray, shape (batch_size, ), node ids
        :param node_interact_times: ndarray, shape (batch_size, ), node interaction times
        :return:
        """
        # three lists to store the first-hop neighbor ids, edge ids and interaction timestamp information, with batch_size as the list length
        nodes_neighbor_ids_list, nodes_edge_ids_list, nodes_neighbor_times_list = [], [], []
        # get the temporal neighbors at the first hop
        for idx, (node_id, node_interact_time) in enumerate(zip(node_ids, node_interact_times)):
            # find neighbors that interacted with node_id before time node_interact_time
            node_neighbor_ids, node_edge_ids, node_neighbor_times, _ = self.find_neighbors_before(node_id=node_id,
                                                                                                  interact_time=node_interact_time,
                                                                                                  return_sampled_probabilities=False)
            nodes_neighbor_ids_list.append(node_neighbor_ids)
            nodes_edge_ids_list.append(node_edge_ids)
            nodes_neighbor_times_list.append(node_neighbor_times)

        return nodes_neighbor_ids_list, nodes_edge_ids_list, nodes_neighbor_times_list
    
    def reset_random_state(self):
        """
        reset the random state by self.seed
        :return:
        """
        self.random_state = np.random.RandomState(self.seed)

def str2dict(s):
    # accepts a str like " 'k1':v1; ...; 'km':vm ", values (v1,...,vm) can be single values or lists (for hyperparameter tuning)
    output = dict()
    kv_pairs = s.replace(' ','').replace("'",'').split(';')
    for kv in kv_pairs:
        key = kv.split(':')[0]
        v_ = kv.split(':')[1]
        if '[' in v_:
            # transform list of values
            v_list = v_.replace('[','').replace(']','').split(',')
            vs=[]
            for v__ in v_list:
                try:
                    # if the parameter is float
                    vs.append(float(v__))
                except:
                    # if the parameter is str
                    vs.append(str(v__))
            output.update({key:vs})
        else:
            try:
                output.update({key: float(v_)})
            except:
                output.update({key: str(v_)})
    return output





def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

def calculate_avg_performance_forgetting(task_acc_vary_cur,metric,num_tasks):
    # num_tasks = len(task_acc_vary_cur)
    num_metrics = len(task_acc_vary_cur[0][0])

    # Initialize dictionaries to store performance and forgetting for each metric
    avg_performance = {f'{i}': [] for i in metric}
    all_forgetting = {f'{i}': [] for i in metric}
    all_forgetting_2 = {f'{i}': [] for i in metric}
    all_BWT = {f'{i}': [] for i in metric}
    all_forgetting_max = {f'{i}': [] for i in metric}
    only_forgetting = {f'{i}': [] for i in metric}

    # Compute the average performance for each metric
    for metric_idx in range(len(metric)):
        metric_avg_performance = []
        for j in range(num_tasks):
            col_sum = sum(task_acc_vary_cur[i][j][metric_idx] for i in range(j + 1))
            metric_avg_performance.append(col_sum / (j + 1))
        
        avg_performance[metric[metric_idx]] = metric_avg_performance

    # Compute the forgetting for each task and each metric
    for metric_idx in range(len(metric)):
        only_f = []
        for i in range(1, num_tasks):
            of = 0
            for j in range(i):
                only_forg = task_acc_vary_cur[j][j][metric_idx] - task_acc_vary_cur[j][i][metric_idx]
                if only_forg > 0:
                    of += only_forg
            only_f.append(of / i)
        
        only_forgetting[metric[metric_idx]] = only_f

    for metric_idx in range(len(metric)):
        metric_forgetting = []
        for i in range(1, num_tasks):
            f = 0
            for j in range(i):
                forgetting = task_acc_vary_cur[j][j][metric_idx] - task_acc_vary_cur[j][i][metric_idx]
                f += max(forgetting,0)
            metric_forgetting.append(f / i)
        
        all_forgetting[metric[metric_idx]] = metric_forgetting

    for metric_idx in range(len(metric)):
        metric_BWT = []
        for i in range(1, num_tasks):
            b = 0
            for j in range(i):
                bwt =task_acc_vary_cur[j][i][metric_idx] - task_acc_vary_cur[j][j][metric_idx]
                b += bwt
            metric_BWT.append(b / i)
        
        all_BWT[metric[metric_idx]] = metric_BWT

   
    max_performance = [[[0,0,0,0] for _ in range(num_tasks)] for _ in range(num_tasks)]
    # print(task_acc_vary_cur)

   
    for metric_idx in range(len(metric)):
        for i in range(num_tasks):
            for j in range(i,num_tasks):
                if j-1 == 0:
                    max_performance[i][j][metric_idx] = task_acc_vary_cur[i][j-1][metric_idx]
                else:
                    a = [task_acc_vary_cur[i][t][metric_idx] for t in range(i,j)]
                    if a:
                        # max_performance[i][j] = max(task_acc_vary_cur[i][j][metric_idx],  max_performance[i][j])
                        max_performance[i][j][metric_idx] = max(a)
                    else:
                        max_performance[i][j][metric_idx] = 0
                    
    # print('max_performance',max_performance)

    for metric_idx, metric_name in enumerate(metric):
        metric_forgetting_max = []
        for k in range(1, num_tasks):
            fm = 0
            for j in range(k):
                fmm = max_performance[j][k][metric_idx] - task_acc_vary_cur[j][k][metric_idx]
                # forgetting_m = max(0, fmm)  # Forgetting should not be negative
                fm += fmm
            # Average forgetting for task k
            af_max = fm / k
            metric_forgetting_max.append(af_max)
        all_forgetting_max[metric_name] = metric_forgetting_max
        

    return avg_performance, all_forgetting, all_BWT,all_forgetting_max,only_forgetting

def average_result(dict_list):
    num_dicts = len(dict_list)
    if num_dicts == 0:
        return {}

    # Initialize a dictionary to store the sum of all values
    sum_dict = {key: [0] * len(values) for key, values in dict_list[0].items()}

    # Sum all values from each dictionary
    for d in dict_list:
        for key, values in d.items():
            sum_dict[key] = [sum_dict[key][i] + values[i] for i in range(len(values))]

    # Calculate the average
    avg_dict = {key: [val / num_dicts for val in values] for key, values in sum_dict.items()}

    return avg_dict
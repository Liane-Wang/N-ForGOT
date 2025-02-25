import numpy as np
import pandas as pd
import random
from copy import deepcopy

# import torch
# import torch.nn.functional as F


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
    def normalize_timestamps(self,min_timestamp):
        self.timestamps -= min_timestamp

def count_labels_in_task(task):
    labels, counts = np.unique(task.labels_dst, return_counts=True)
    return dict(zip(labels, counts))

def get_data(DatasetName,n_task, n_class, blurry,fuzzy_boundary):
    if DatasetName == "yelp":
        start_y = 2014
        end_y = start_y + n_task -1
        graph = pd.read_csv("./data/new_{}.csv".format(DatasetName))
        node_features = np.load("./data/new_{}_node.npy".format(DatasetName))
        src = graph.u.values
        dst = graph.i.values
        edge_idxs = graph.idx.values
        labels_src = graph.label_src.values
        labels_dst = graph.label_dst.values
        timestamps = graph.timestamp.values
        task = graph.year.values
    else: 
        graph = pd.read_csv("./data/{}.csv".format(DatasetName))
        node_features = np.load("./data/{}_node.npy".format(DatasetName))
    # edge_features = np.load("./data/{}.npy".format(DatasetName))
        src = graph.u.values
        dst = graph.i.values
        edge_idxs = graph.idx.values
        labels_src = graph.label_u.values
        labels_dst = graph.label_i.values
        timestamps = graph.ts.values
        # task = graph.year.values
    timestamps = timestamps - timestamps[0]  ##
    n_task = n_task
    
    all_data = Data(src, dst, timestamps, edge_idxs, labels_src, labels_dst)
    edge_features = np.zeros((len(edge_idxs), node_features.shape[1]))
    # edge_features = np.array([[0] * 300 for i in range(len(edge_idxs))])
    
    node_set = set(src) | set(dst)

    print(
        "The dataset has {} interactions, involving {} different nodes".format(
            len(src), len(node_set)
        )
    )

    task_full_mask = [[False] * len(src) for i in range(n_task)]
    tmp = 0

    heter = 0
    node_first_time = {}
    cos = 0
    for i in range(len(src)): 
        if task[i] > end_y: 
            
            break
        if fuzzy_boundary:
            tmp = task[i]-start_y
            task_full_mask[tmp][i] = True
            
        else:
            # mx_label = max(labels_src[i], labels_dst[i])
            # # if blurry:
            # #     tmp = max(tmp, mx_label // n_class)
            # # else:
            # tmp = mx_label // n_class
            src_label = labels_src[i]
            dst_label = labels_dst[i]
            tmp_src = src_label // n_class
            tmp_dst = dst_label // n_class
            if tmp_src == task[i]-start_y and tmp_dst == task[i]-start_y:
                task_full_mask[tmp_src][i] = True
                task_full_mask[tmp_dst][i] = True

        cur_time = timestamps[i]
        if src[i] not in node_first_time:
            node_first_time[src[i]] = cur_time
        if dst[i] not in node_first_time:
            node_first_time[dst[i]] = cur_time

    full_data = [
        Data(
            src[task_full_mask[i]],
            dst[task_full_mask[i]],
            timestamps[task_full_mask[i]],
            edge_idxs[task_full_mask[i]],
            labels_src[task_full_mask[i]],
            labels_dst[task_full_mask[i]],
        )
        for i in range(n_task)
    ]
   
    task_full_node_set = [
        set(src[task_full_mask[i]]) | set(dst[task_full_mask[i]]) for i in range(n_task)
    ]

    random.seed(42)

    train_mask = [[False] * len(src) for i in range(n_task)]
    val_mask = [[False] * len(src) for i in range(n_task)]
    test_mask = [[False] * len(src) for i in range(n_task)]

    train_data = []
    val_data = []
    test_data = []

    re_train_data = []
    re_val_data = []

    for i in range(n_task):
        tmp_train_node_set = set(
            random.sample(task_full_node_set[i], int(0.8 * len(task_full_node_set[i])))
        )
        tmp_no_train_node_set = set(task_full_node_set[i]) - tmp_train_node_set
        tmp_val_node_set = set(
            random.sample(tmp_no_train_node_set, int(0.5 * len(tmp_no_train_node_set)))
        )
        tmp_test_node_set = tmp_no_train_node_set - tmp_val_node_set

        tmp_train_mask = [
            (src[j] in tmp_train_node_set or dst[j] in tmp_train_node_set)
            for j in range(len(src))
        ]
        tmp_val_mask = [
            (src[j] in tmp_val_node_set or dst[j] in tmp_val_node_set)
            for j in range(len(src))
        ]
        tmp_test_mask = [
            (src[j] in tmp_test_node_set or dst[j] in tmp_test_node_set)
            for j in range(len(src))
        ]

        tmp_no_train_src_mask = graph.u.map(lambda x: x in tmp_no_train_node_set).values
        tmp_no_train_dst_mask = graph.i.map(lambda x: x in tmp_no_train_node_set).values
        tmp_observed_edges_mask = np.logical_and(
            ~tmp_no_train_src_mask, ~tmp_no_train_dst_mask
        )

        train_mask[i] = np.logical_and(tmp_train_mask, tmp_observed_edges_mask)
        train_mask[i] = np.logical_and(train_mask[i], task_full_mask[i])
        val_mask[i] = np.logical_and(tmp_val_mask, task_full_mask[i])
        test_mask[i] = np.logical_and(tmp_test_mask, task_full_mask[i])

        train_data.append(
            Data(
                src[train_mask[i]],
                dst[train_mask[i]],
                timestamps[train_mask[i]],
                edge_idxs[train_mask[i]],
                labels_src[train_mask[i]],
                labels_dst[train_mask[i]],
            )
        )

        val_data.append(
            Data(
                src[val_mask[i]],
                dst[val_mask[i]],
                timestamps[val_mask[i]],
                edge_idxs[val_mask[i]],
                labels_src[val_mask[i]],
                labels_dst[val_mask[i]],
            )
        )

        test_data.append(
            Data(
                src[test_mask[i]],
                dst[test_mask[i]],
                timestamps[test_mask[i]],
                edge_idxs[test_mask[i]],
                labels_src[test_mask[i]],
                labels_dst[test_mask[i]],
            )
        )

        val_data[-1].induct_nodes = (
            val_data[-1].unique_nodes
            - train_data[-1].unique_nodes
            - test_data[-1].unique_nodes
        )
        test_data[-1].induct_nodes = (
            test_data[-1].unique_nodes
            - train_data[-1].unique_nodes
            - val_data[-1].unique_nodes
        )
        print("Task", i, end=" ### ")
        print(
            "unique nodes:",
            "full",
            full_data[i].n_unique_nodes,
            "train",
            train_data[i].n_unique_nodes,
            "val",
            val_data[i].n_unique_nodes,
            "test",
            test_data[i].n_unique_nodes,
            "###",
            "interactions:",
            "full",
            full_data[i].n_interactions,
            "train",
            train_data[i].n_interactions,
            "val",
            val_data[i].n_interactions,
            "test",
            test_data[i].n_interactions,
        )

        if i == 0:
            re_train_data.append(train_data[i])
            re_val_data.append(val_data[i])
        else:
            re_train_data.append(deepcopy(re_train_data[i - 1]))
            re_val_data.append(deepcopy(re_val_data[i - 1]))
            re_train_data[i].add_data(train_data[i])
            re_val_data[i].add_data(val_data[i])

        # print(len(re_train_data[i].src))
        re_val_data[-1].induct_nodes = (
            re_val_data[-1].unique_nodes - re_train_data[-1].unique_nodes
        )


    label_counts_per_task_train = [count_labels_in_task(task) for task in train_data]
    print("interactions of class in training set:" )
    for i, label_counts in enumerate(label_counts_per_task_train):
        print(f"Task {i}: {label_counts}")

    label_counts_per_task_test = [count_labels_in_task(task) for task in test_data]
    print("interactions of class in testing set:" )
    for i, label_counts in enumerate(label_counts_per_task_test):
        print(f"Task {i}: {label_counts}")
    

    return (
        node_features,
        edge_features,
        full_data,
        train_data,
        val_data,
        test_data,
        all_data,
        re_train_data,
        re_val_data,
    )


def get_data_old(DatasetName, n_task, n_class, blurry):
    graph = pd.read_csv("./data/{}.csv".format(DatasetName))
    # edge_features = np.load("./data/{}.npy".format(DatasetName))
    node_features = np.load("./data/{}_node.npy".format(DatasetName), allow_pickle=True)

    src = graph.u.values
    dst = graph.i.values
    edge_idxs = graph.idx.values
    labels_src = graph.label_u.values
    labels_dst = graph.label_i.values
    timestamps = graph.ts.values
    timestamps = timestamps - timestamps[0] + 1
  
    all_data = Data(src, dst, timestamps, edge_idxs, labels_src, labels_dst)
    edge_features = np.zeros((len(edge_idxs), node_features.shape[1]))
    # edge_features = np.array([[0] * 300 for i in range(len(edge_idxs))])

    node_set = set(src) | set(dst)

    print(
        "The dataset has {} interactions, involving {} different nodes".format(
            len(src), len(node_set)
        )
    )

    task_full_mask = [[False] * len(src) for i in range(n_task)]
    tmp = 0

    heter = 0
    node_first_time = {}
    cos = 0
    for i in range(len(src)):
        mx_label = max(labels_src[i], labels_dst[i])
        if blurry:
            tmp = max(tmp, mx_label // n_class)
        else:
            tmp = mx_label // n_class
        if tmp >= n_task:
            break
        task_full_mask[tmp][i] = True
        cur_time = timestamps[i]
        if src[i] not in node_first_time:
            node_first_time[src[i]] = cur_time
        if dst[i] not in node_first_time:
            node_first_time[dst[i]] = cur_time

    full_data = [
        Data(
            src[task_full_mask[i]],
            dst[task_full_mask[i]],
            timestamps[task_full_mask[i]],
            edge_idxs[task_full_mask[i]],
            labels_src[task_full_mask[i]],
            labels_dst[task_full_mask[i]],
        )
        for i in range(n_task)
    ]

    task_full_node_set = [
        set(src[task_full_mask[i]]) | set(dst[task_full_mask[i]]) for i in range(n_task)
    ]

    random.seed(42)

    train_mask = [[False] * len(src) for i in range(n_task)]
    val_mask = [[False] * len(src) for i in range(n_task)]
    test_mask = [[False] * len(src) for i in range(n_task)]

    train_data = []
    val_data = []
    test_data = []

    re_train_data = []
    re_val_data = []

    for i in range(n_task):
        tmp_train_node_set = set(
            random.sample(task_full_node_set[i], int(0.8 * len(task_full_node_set[i])))
        )
        tmp_no_train_node_set = set(task_full_node_set[i]) - tmp_train_node_set
        tmp_val_node_set = set(
            random.sample(tmp_no_train_node_set, int(0.5 * len(tmp_no_train_node_set)))
        )
        tmp_test_node_set = tmp_no_train_node_set - tmp_val_node_set

        tmp_train_mask = [
            (src[j] in tmp_train_node_set or dst[j] in tmp_train_node_set)
            for j in range(len(src))
        ]
        tmp_val_mask = [
            (src[j] in tmp_val_node_set or dst[j] in tmp_val_node_set)
            for j in range(len(src))
        ]
        tmp_test_mask = [
            (src[j] in tmp_test_node_set or dst[j] in tmp_test_node_set)
            for j in range(len(src))
        ]

        tmp_no_train_src_mask = graph.u.map(lambda x: x in tmp_no_train_node_set).values
        tmp_no_train_dst_mask = graph.i.map(lambda x: x in tmp_no_train_node_set).values
        tmp_observed_edges_mask = np.logical_and(
            ~tmp_no_train_src_mask, ~tmp_no_train_dst_mask
        )

        train_mask[i] = np.logical_and(tmp_train_mask, tmp_observed_edges_mask)
        train_mask[i] = np.logical_and(train_mask[i], task_full_mask[i])
        val_mask[i] = np.logical_and(tmp_val_mask, task_full_mask[i])
        test_mask[i] = np.logical_and(tmp_test_mask, task_full_mask[i])

        train_data.append(
            Data(
                src[train_mask[i]],
                dst[train_mask[i]],
                timestamps[train_mask[i]],
                edge_idxs[train_mask[i]],
                labels_src[train_mask[i]],
                labels_dst[train_mask[i]],
            )
        )

        val_data.append(
            Data(
                src[val_mask[i]],
                dst[val_mask[i]],
                timestamps[val_mask[i]],
                edge_idxs[val_mask[i]],
                labels_src[val_mask[i]],
                labels_dst[val_mask[i]],
            )
        )

        test_data.append(
            Data(
                src[test_mask[i]],
                dst[test_mask[i]],
                timestamps[test_mask[i]],
                edge_idxs[test_mask[i]],
                labels_src[test_mask[i]],
                labels_dst[test_mask[i]],
            )
        )

        val_data[-1].induct_nodes = (
            val_data[-1].unique_nodes
            - train_data[-1].unique_nodes
            - test_data[-1].unique_nodes
        )
        test_data[-1].induct_nodes = (
            test_data[-1].unique_nodes
            - train_data[-1].unique_nodes
            - val_data[-1].unique_nodes
        )
        print("Task", i, end=" ### ")
        print(
            "unique nodes:",
            "full",
            full_data[i].n_unique_nodes,
            "train",
            train_data[i].n_unique_nodes,
            "val",
            val_data[i].n_unique_nodes,
            "test",
            test_data[i].n_unique_nodes,
            "###",
            "interactions:",
            "full",
            full_data[i].n_interactions,
            "train",
            train_data[i].n_interactions,
            "val",
            val_data[i].n_interactions,
            "test",
            test_data[i].n_interactions,
        )

        if i == 0:
            re_train_data.append(train_data[i])
            re_val_data.append(val_data[i])
        else:
            re_train_data.append(deepcopy(re_train_data[i - 1]))
            re_val_data.append(deepcopy(re_val_data[i - 1]))
            re_train_data[i].add_data(train_data[i])
            re_val_data[i].add_data(val_data[i])

        # print(len(re_train_data[i].src))
        re_val_data[-1].induct_nodes = (
            re_val_data[-1].unique_nodes - re_train_data[-1].unique_nodes
        )

   

    # label_counts_per_task_train = [count_labels_in_task(task) for task in train_data]
    # print("interactions of class in training set:" )
    # for i, label_counts in enumerate(label_counts_per_task_train):
    #     print(f"Task {i}: {label_counts}")

    # label_counts_per_task_test = [count_labels_in_task(task) for task in test_data]
    # print("interactions of class in testing set:" )
    # for i, label_counts in enumerate(label_counts_per_task_test):
    #     print(f"Task {i}: {label_counts}")
    
    return (
        node_features,
        edge_features,
        full_data,
        train_data,
        val_data,
        test_data,
        all_data,
        re_train_data,
        re_val_data,
    )


def computer_time_statics(src, dst, timestamps):
    last_timestamp_src = dict()
    last_timestamp_dst = dict()
    all_timediffs_src = []
    all_timediffs_dst = []
    for k in range(len(src)):
        src_id = src[k]
        dst_id = dst[k]
        cur_timestamp = timestamps[k]
        if src_id not in last_timestamp_src.keys():
            last_timestamp_src[src_id] = 0
        if dst_id not in last_timestamp_dst.keys():
            last_timestamp_dst[dst_id] = 0
        all_timediffs_src.append(cur_timestamp - last_timestamp_src[src_id])
        all_timediffs_dst.append(cur_timestamp - last_timestamp_dst[dst_id])
        last_timestamp_src[src_id] = cur_timestamp
        last_timestamp_dst[dst_id] = cur_timestamp
    assert len(all_timediffs_src) == len(src)
    assert len(all_timediffs_dst) == len(dst)
    mean_time_shift_src = np.mean(all_timediffs_src)
    std_time_shift_src = np.std(all_timediffs_src)
    mean_time_shift_dst = np.mean(all_timediffs_dst)
    std_time_shift_dst = np.std(all_timediffs_dst)
    return (
        mean_time_shift_src,
        std_time_shift_src,
        mean_time_shift_dst,
        std_time_shift_dst,
    )

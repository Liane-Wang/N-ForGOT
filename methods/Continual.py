import torch
from torch import nn
from models.Backbone import TemporalGNNClassifier

# The following code is to initialize the class for implementing baseline methods in continual learning. 
# Please generate a template code for methods in continual learning by using the following code.
class Continual(nn.Module):
    def __init__(self, args, neighbor_finder, node_features, edge_features, src_label, dst_label):
        super(Continual, self).__init__()
        self.args = args

        if args.supervision == 'supervised':
            self.model = TemporalGNNClassifier(args, neighbor_finder, node_features, edge_features, src_label, dst_label)
        
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

        data_dict = {}

        if self.args.supervision == 'supervised':
            loss = self.model(src_nodes, dst_nodes, edges, edge_times
                                , n_neighbors, dataset_idx)
            data_dict[ 'loss' ] = loss.item()
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


    def get_acc(self, x, y):
        output = self.model(x)
        _, pred = torch.max(output, 1)
        correct = (pred == y).sum().item()
        return correct / len(y)

    def get_logits(self, src_nodes, dst_nodes, edges, edge_times, n_neighbors, dataset_idx):
        src_logits, dst_logits = self.model(src_nodes, dst_nodes, edges, edge_times, n_neighbors, dataset_idx, return_logits=True)
        return src_logits, dst_logits
    
    def end_dataset(self, train_data, args):
        return

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


# Path: tgcl/dev_tgcl-main/SubGraphCL/methods/Finetune.py
# Compare this snippet from tgcl/dev_tgcl-main/SubGraphCL/methods/Continual.py:
# import torch
# from torch import nn
#   
# from models.Backbone import TemporalGNNClassifier
#   
# # The following code is to initialize the class for implementing baseline methods in continual learning such as lef and ewc.
# # Please generate a template code for methods in continual learning by using the following code.
   
# class Continual(nn.Module):
#     def __init__(self, args, neighbor_finder, node_features, edge_features, src_label, dst_label):
#         super(Continual, self).__init__()
#         self.args = args
#   

#         if args.supervision == 'supervised':
#             self.model = TemporalGNNClassifier(args, neighbor_finder, node_features, edge_features, src_label, dst_label)
#
#         self.optimizer = torch.optim.Adam(self.model.parameters(), lr=args.lr)

import torch
from torch import nn
import numpy as np

class Memory(nn.Module):
  def __init__(self, n_nodes, emb_dim, device='cpu', features=None):
    super(Memory, self).__init__()
    self.n_nodes = n_nodes
    self.emb_dim = emb_dim
    self.device = device
    self.features=features
    self.__init_memory__()

  def __init_memory__(self, nodes=None, seed=0):
    seed=0
    torch.manual_seed(seed)
    if nodes is not None:
        tmp = nn.Parameter(torch.zeros((self.n_nodes+1, self.emb_dim)).to(self.device), requires_grad=False)
        nn.init.xavier_normal_(tmp)
        self.emb.detach_()
        self.emb[list(nodes)] = tmp[list(nodes)]
    else:    
        if self.features is None:
            self.emb = nn.Parameter(torch.zeros((self.n_nodes+1, self.emb_dim)).to(self.device), requires_grad=False)
            nn.init.xavier_normal_(self.emb)
            self.emb[0] = 0.0
        else:
            self.emb = nn.Parameter(torch.zeros((self.features.shape[0]+1,self.features.shape[1])).float().to(self.device), requires_grad=False)
            self.emb[0] = 0.0
            self.emb[1:] = nn.Parameter(torch.tensor(self.features).float().to(self.device), requires_grad=False) 

  def detach_memory(self):
    self.emb.detach_()


class LinearTimeMMDLoss(torch.nn.Module):
    def __init__(self, kernel_mul=2.0, kernel_num=5):
        super(LinearTimeMMDLoss, self).__init__()
        self.kernel_mul = kernel_mul
        self.kernel_num = kernel_num

    def gaussian_kernel(self, x1, x2, y1, y2, kernel_mul=2.0, kernel_num=5):
        # Calculate the Gaussian kernel values for the pairs (x1, x2) and (y1, y2)
        L2_distance_xx = torch.sum((x1 - x2) ** 2)
        L2_distance_yy = torch.sum((y1 - y2) ** 2)
        L2_distance_xy = torch.sum((x1 - y2) ** 2)
        L2_distance_yx = torch.sum((x2 - y1) ** 2)

        bandwidth = (L2_distance_xx + L2_distance_yy + L2_distance_xy + L2_distance_yx) / 4
        bandwidth /= kernel_mul ** (kernel_num // 2)
        bandwidth_list = [bandwidth * (kernel_mul ** i) for i in range(kernel_num)]

        kernel_val_xx = [torch.exp(-L2_distance_xx / b) for b in bandwidth_list]
        kernel_val_yy = [torch.exp(-L2_distance_yy / b) for b in bandwidth_list]
        kernel_val_xy = [torch.exp(-L2_distance_xy / b) for b in bandwidth_list]
        kernel_val_yx = [torch.exp(-L2_distance_yx / b) for b in bandwidth_list]

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

        # Compute h for each pair
        h_values = torch.stack([
            self.gaussian_kernel(x_odd[i], x_even[i], y_odd[i], y_even[i], self.kernel_mul, self.kernel_num)
            for i in range(m2)
        ])

        return h_values.mean()

# # Example usage
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# source = torch.randn(100, 10).to(device)  # Example source domain data
# target = torch.randn(100, 10).to(device)  # Example target domain data

# mmd_loss = LinearTimeMMDLoss().to(device)
# mmd_value = mmd_loss(source, target)
# print("MMD l2 value:", mmd_value.item())

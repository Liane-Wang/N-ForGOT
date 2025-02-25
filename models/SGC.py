from .gnnconv import GATConv, GCNLayer, GINConv
from .layers import PairNorm
from .utils import *
from dgl.base import DGLError
from dgl.nn.pytorch import edge_softmax
import dgl.function as fn
import torch.nn as nn
import torch as th
linear_choices = {'nn.Linear':nn.Linear, 'Linear_IL':Linear_IL}

class SGC_Agg(nn.Module):
    # only the neighborhood aggregation of SGC
    def __init__(self, k=1, cached=False, norm=None, allow_zero_in_degree=False):
        super().__init__()
        self._cached = cached
        self._cached_h = None
        self._k = k
        self.norm = norm
        self._allow_zero_in_degree = allow_zero_in_degree

    def set_allow_zero_in_degree(self, set_value):
        r"""

        Description
        -----------
        Set allow_zero_in_degree flag.

        Parameters
        ----------
        set_value : bool
            The value to be set to the flag.
        """
        self._allow_zero_in_degree = set_value

    def forward(self, graph, feat):
        r"""

        Description
        -----------
        Compute Simplifying Graph Convolution layer.

        Parameters
        ----------
        graph : DGLGraph
            The graph.
        feat : torch.Tensor
            The input feature of shape :math:`(N, D_{in})` where :math:`D_{in}`
            is size of input feature, :math:`N` is the number of nodes.

        Returns
        -------
        torch.Tensor
            The output feature of shape :math:`(N, D_{out})` where :math:`D_{out}`
            is size of output feature.

        Raises
        ------
        DGLError
            If there are 0-in-degree nodes in the input graph, it will raise DGLError
            since no message will be passed to those nodes. This will cause invalid output.
            The error can be ignored by setting ``allow_zero_in_degree`` parameter to ``True``.

        Note
        ----
        If ``cache`` is set to True, ``feat`` and ``graph`` should not change during
        training, or you will get wrong results.
        """
        with graph.local_scope():
            if not self._allow_zero_in_degree:
                if (graph.in_degrees() == 0).any():
                    raise DGLError('There are 0-in-degree nodes in the graph, '
                                   'output for those nodes will be invalid. '
                                   'This is harmful for some applications, '
                                   'causing silent performance regression. '
                                   'Adding self-loop on the input graph by '
                                   'calling `g = dgl.add_self_loop(g)` will resolve '
                                   'the issue. Setting ``allow_zero_in_degree`` '
                                   'to be `True` when constructing this module will '
                                   'suppress the check and let the code run.')

            if self._cached_h is not None:
                feat = self._cached_h
            else:
                # compute normalization
                degs = graph.in_degrees().float().clamp(min=1)
                norm = th.pow(degs, -0.5)
                norm = norm.to(feat.device).unsqueeze(1)
                # compute (D^-1 A^k D)^k X
                for _ in range(self._k):
                    feat = feat * norm
                    graph.ndata['h'] = feat
                    graph.update_all(fn.copy_u('h', 'm'), fn.sum('m', 'h'))
                    feat = graph.ndata.pop('h')
                    feat = feat * norm

                if self.norm is not None:
                    feat = self.norm(feat)

                # cache feature
                if self._cached:
                    self._cached_h = feat
            # return self.fc(feat)
            return feat

    def forward_batch(self, blocks, feat):
        r"""

        Description
        -----------
        Compute Simplifying Graph Convolution layer.

        Parameters
        ----------
        graph : DGLGraph
            The graph.
        feat : torch.Tensor
            The input feature of shape :math:`(N, D_{in})` where :math:`D_{in}`
            is size of input feature, :math:`N` is the number of nodes.

        Returns
        -------
        torch.Tensor
            The output feature of shape :math:`(N, D_{out})` where :math:`D_{out}`
            is size of output feature.

        Raises
        ------
        DGLError
            If there are 0-in-degree nodes in the input graph, it will raise DGLError
            since no message will be passed to those nodes. This will cause invalid output.
            The error can be ignored by setting ``allow_zero_in_degree`` parameter to ``True``.

        Note
        ----
        If ``cache`` is set to True, ``feat`` and ``graph`` should not change during
        training, or you will get wrong results.
        """
        if self._k != len(blocks):
            raise DGLError('The depth of the dataloader sampler is incompatible with the depth of SGC')
        for block in blocks:
            with block.local_scope():
                if not self._allow_zero_in_degree:
                    if (block.in_degrees() == 0).any():
                        raise DGLError('There are 0-in-degree nodes in the graph, '
                                       'output for those nodes will be invalid. '
                                       'This is harmful for some applications, '
                                       'causing silent performance regression. '
                                       'Adding self-loop on the input graph by '
                                       'calling `g = dgl.add_self_loop(g)` will resolve '
                                       'the issue. Setting ``allow_zero_in_degree`` '
                                       'to be `True` when constructing this module will '
                                       'suppress the check and let the code run.')

                if self._cached_h is not None:
                    feat = self._cached_h
                else:
                    # compute normalization
                    degs = block.out_degrees().float().clamp(min=1)
                    norm = th.pow(degs, -0.5)
                    norm = norm.to(feat.device).unsqueeze(1)
                    # compute (D^-1 A^k D)^k X
                    feat = feat * norm
                    block.srcdata['h'] = feat
                    block.update_all(fn.copy_u('h', 'm'), fn.sum('m', 'h'))
                    feat = block.dstdata.pop('h')
                    degs = block.in_degrees().float().clamp(min=1)
                    norm = th.pow(degs, -0.5)
                    norm = norm.to(feat.device).unsqueeze(1)
                    feat = feat * norm

        with blocks[-1].local_scope():
            if self.norm is not None:
                feat = self.norm(feat)

            # cache feature
            if self._cached:
                self._cached_h = feat

        # return self.fc(feat)
        return feat

class SGC(nn.Module):
    def __init__(self, args):
        super(SGC, self).__init__()
        linear_layer = linear_choices[args.SGC_args['linear']]
        if args.method == 'twp':
            self.twp=True
        else:
            self.twp=False
        self.bn = args.SGC_args['batch_norm']
        self.dropout = args.SGC_args['dropout']
        self.leaky_relu = nn.LeakyReLU(0.2)
        self.gpu = args.gpu
        self.neighbor_agg = SGC_Agg(k=args.SGC_args['k'])
        self.feat_trans_layers = nn.ModuleList()
        if self.bn:
            self.bns = nn.ModuleList()
        h_dims = args.SGC_args['h_dims']
        if len(h_dims) > 0:
            self.feat_trans_layers.append(linear_layer(args.d_data, h_dims[0], bias=args.SGC_args['linear_bias']))
            if self.bn:
                self.bns.append(nn.BatchNorm1d(h_dims[0]))
            for i in range(len(h_dims) - 1):
                self.feat_trans_layers.append(linear_layer(h_dims[i], h_dims[i + 1], bias=args.SGC_args['linear_bias']))
                if self.bn:
                    self.bns.append(nn.BatchNorm1d(h_dims[i + 1]))
            self.feat_trans_layers.append(linear_layer(h_dims[-1], args.n_cls, bias=args.SGC_args['linear_bias']))
        elif len(h_dims) == 0:
            self.feat_trans_layers.append(linear_layer(args.d_data, args.n_cls, bias=args.SGC_args['linear_bias']))
        else:
            raise ValueError('no valid MLP dims are given')

    def forward(self, graph, x, twp=False, tasks=None):
        graph = graph.local_var().to('cuda:{}'.format(self.gpu))
        e_list = []
        x = self.neighbor_agg(graph, x)
        logits, e = self.feat_trans(graph, x, twp=twp, cls=tasks)
        e_list = e_list + e
        return logits, e_list

    def forward_batch(self, blocks, x, twp=False, tasks=None):
        #graph = graph.local_var().to('cuda:{}'.format(self.gpu))
        e_list = []
        x = self.neighbor_agg.forward_batch(blocks, x)
        logits, e = self.feat_trans(blocks[0], x, twp=twp, cls=tasks)
        e_list = e_list + e
        return logits, e_list

    def feat_trans(self, graph, x, twp=False, cls=None):
        for i, layer in enumerate(self.feat_trans_layers[:-1]):
            x = layer(x)
            if self.bn:
                x = self.bns[i](x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.feat_trans_layers[-1](x)

        self.second_last_h = x

        mask = torch.zeros(x.shape[-1], device=x.get_device())
        if cls is not None:
            mask[cls] = 1.
        else:
            mask[:] = 1.
        x = x * mask
        # for twp
        elist = []
        if self.twp:
            graph.srcdata['h'] = x
            graph.apply_edges(
                lambda edges: {'e': torch.sum((torch.mul(edges.src['h'], torch.tanh(edges.dst['h']))), 1)})
            e = self.leaky_relu(graph.edata.pop('e'))
            e_soft = edge_softmax(graph, e)

            elist.append(e_soft)

        return x, elist
        #return x.log_softmax(dim=-1), elist
    def reset_params(self):
        for layer in self.feat_trans_layers:
            layer.reset_parameters()
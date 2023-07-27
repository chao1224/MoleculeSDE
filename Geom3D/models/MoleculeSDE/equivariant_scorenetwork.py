from typing import Union
from torch_geometric.typing import OptPairTensor, Adj, OptTensor, Size

import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F
from torch_sparse import SparseTensor
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.nn import TransformerConv


class GATLayer(nn.Module):
    def __init__(self, n_head, hidden_dim, dropout=0.2):
        super(GATLayer, self).__init__()

        assert hidden_dim % n_head == 0
        self.MHA = TransformerConv(
            in_channels=hidden_dim,
            out_channels=int(hidden_dim // n_head),
            heads=n_head,
            dropout=dropout,
            edge_dim=hidden_dim,
        )
        self.FFN = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim)
        )
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)

    def forward(self, edge_index, node_attr, edge_attr):
        x = self.MHA(node_attr, edge_index, edge_attr)
        node_attr = node_attr + self.norm1(x)
        x = self.FFN(node_attr)
        node_attr = node_attr + self.norm2(x)
        
        return node_attr


class EquiLayer(MessagePassing):
    def __init__(self, eps=0., train_eps=False, activation="silu", **kwargs):
        super(EquiLayer, self).__init__(aggr='mean', **kwargs)
        self.initial_eps = eps

        if isinstance(activation, str):
            self.activation = getattr(F, activation)
        else:
            self.activation = None   

        if train_eps:
            self.eps = torch.nn.Parameter(torch.Tensor([eps]))
        else:
            self.register_buffer('eps', torch.Tensor([eps]))

    def forward(self, x: Union[Tensor, OptPairTensor], edge_index: Adj,
                edge_attr: OptTensor = None, size: Size = None) -> Tensor:
        if isinstance(x, Tensor):
            x: OptPairTensor = (x, x)

        # Node and edge feature dimensionalites need to match.
        if isinstance(edge_index, Tensor):
            assert edge_attr is not None
            # assert x[0].size(-1) == edge_attr.size(-1)
        elif isinstance(edge_index, SparseTensor):
            assert x[0].size(-1) == edge_index.size(-1)

        # propagate_type: (x: OptPairTensor, edge_attr: OptTensor)
        out = self.propagate(edge_index, x=x, edge_attr=edge_attr, size=size)
        return out

    def message(self, x_j: Tensor, edge_attr: Tensor) -> Tensor: 
        if self.activation:
            return self.activation(x_j + edge_attr)
        else: # TODO: we are mostly using False for activation
            return edge_attr

    def __repr__(self):
        return '{}(nn={})'.format(self.__class__.__name__, self.nn)


class EquivariantScoreNetwork(torch.nn.Module):
    def __init__(self, hidden_dim, hidden_coff_dim=64, activation="silu", short_cut=False, concat_hidden=False):
        super(EquivariantScoreNetwork, self).__init__()

        self.hidden_dim = hidden_dim
        self.num_layers = 2
        self.num_convs = 2
        self.short_cut = short_cut
        self.num_head = 8
        self.dropout = 0.1
        self.concat_hidden = concat_hidden
        self.hidden_coff_dim = hidden_coff_dim

        if isinstance(activation, str):
            self.activation = getattr(F, activation)
        else:
            self.activation = None 
        
        self.gnn_layers = nn.ModuleList()
        self.equi_modules = nn.ModuleList()
        self.basis_mlp_modules = nn.ModuleList()
        for _ in range(self.num_layers):
            trans_convs = nn.ModuleList()
            for _ in range(self.num_convs):
                trans_convs.append(GATLayer(self.num_head, self.hidden_dim, dropout=self.dropout))
            self.gnn_layers.append(trans_convs)

            self.equi_modules.append(EquiLayer(activation=False))

            self.basis_mlp_modules.append(
                nn.Sequential(
                nn.Linear(2 * self.hidden_dim, self.hidden_coff_dim),
                # nn.Softplus(),
                nn.SiLU(),
                nn.Linear(self.hidden_coff_dim, 3))
            )

    def forward(self, edge_index, node_attr, edge_attr, equivariant_basis):
        """
        Args:
            edge_index: edge connection (num_node, 2)
            node_attr: node feature tensor with shape (num_node, hidden)
            edge_attr: edge feature tensor with shape (num_edge, hidden)
            equivariant_basis: an equivariant basis coord_diff, coord_cross, coord_vertical
        Output:
            gradient (score)
        """
        hiddens = []
        conv_input = node_attr # (num_node, hidden)
        coord_diff, coord_cross, coord_vertical = equivariant_basis

        for module_idx, gnn_layers in enumerate(self.gnn_layers):

            for conv_idx, gnn in enumerate(gnn_layers):
                hidden = gnn(edge_index, conv_input, edge_attr)

                if conv_idx < len(gnn_layers) - 1 and self.activation is not None:
                    hidden = self.activation(hidden)
                assert hidden.shape == conv_input.shape                
                if self.short_cut and hidden.shape == conv_input.shape:
                    hidden += conv_input

                hiddens.append(hidden)
                conv_input = hidden

            if self.concat_hidden:
                node_feature = torch.cat(hiddens, dim=-1)
            else:
                node_feature = hiddens[-1]

            h_row, h_col = node_feature[edge_index[0]], node_feature[edge_index[1]] # (num_edge, hidden)
            edge_feature = torch.cat([h_row + h_col, edge_attr], dim=-1)  # (num_edge, 2 * hidden)

            # generate gradient
            dynamic_coff = self.basis_mlp_modules[module_idx](edge_feature)  # (num_edge, 3)
            basis_mix = dynamic_coff[:, :1] * coord_diff + dynamic_coff[:, 1:2] * coord_cross + dynamic_coff[:, 2:3] * coord_vertical  # (num_edge, 3)

            if module_idx == 0:
                gradient = self.equi_modules[module_idx](node_feature, edge_index, basis_mix)
            else:
                gradient += self.equi_modules[module_idx](node_feature, edge_index, basis_mix)

        return {
            "node_feature": node_feature,
            "gradient": gradient
        }


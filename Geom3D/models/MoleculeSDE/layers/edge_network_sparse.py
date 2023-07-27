import math
from typing import Optional, Tuple, Union
import torch
import torch.nn.functional as F
from torch import Tensor
from torch_sparse import SparseTensor
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.nn.dense.linear import Linear
from torch_geometric.typing import Adj, OptTensor, PairTensor

from .common import MultiLayerPerceptron


class EdgeLayer_Tanh(MessagePassing):
    _alpha: OptTensor
    def __init__(
        self,
        in_channels: Union[int, Tuple[int, int]],
        out_channels: int,
        heads: int = 1,
        concat: bool = True,
        beta: bool = False,
        dropout: float = 0.,
        edge_dim: Optional[int] = 1,
        bias: bool = True,
        root_weight: bool = True,
        **kwargs,
    ):
        kwargs.setdefault('aggr', 'mean')
        super(EdgeLayer_Tanh, self).__init__(node_dim=0, **kwargs)

        self.in_channels = in_channels
        self.out_channels = (out_channels//heads)
        self.heads = heads
        self.beta = beta and root_weight
        self.root_weight = root_weight
        self.concat = concat
        self.dropout = dropout
        self.edge_dim = edge_dim
        self._alpha = None

        if isinstance(in_channels, int):
            in_channels = (in_channels, in_channels)

        self.lin_key = Linear(in_channels[0], heads * self.out_channels)
        self.lin_query = Linear(in_channels[1], heads * self.out_channels)
        self.lin_value = Linear(in_channels[0], heads * self.out_channels)
        if edge_dim is not None:
            self.lin_edge = Linear(edge_dim, heads * self.out_channels, bias=False)
        else:
            self.lin_edge = self.register_parameter('lin_edge', None)

        if concat:
            self.lin_skip = Linear(in_channels[1], heads * self.out_channels,
                                   bias=bias)
            if self.beta:
                self.lin_beta = Linear(3 * heads * self.out_channels, 1, bias=False)
            else:
                self.lin_beta = self.register_parameter('lin_beta', None)
        else:
            self.lin_skip = Linear(in_channels[1], self.out_channels, bias=bias)
            if self.beta:
                self.lin_beta = Linear(3 * self.out_channels, 1, bias=False)
            else:
                self.lin_beta = self.register_parameter('lin_beta', None)

        self.reset_parameters()

    def reset_parameters(self):
        self.lin_key.reset_parameters()
        self.lin_query.reset_parameters()
        self.lin_value.reset_parameters()
        if self.edge_dim:
            self.lin_edge.reset_parameters()
        self.lin_skip.reset_parameters()
        if self.beta:
            self.lin_beta.reset_parameters()


    def forward(self, x: Union[Tensor, PairTensor], edge_index: Adj,
                edge_attr: OptTensor = None, return_attention_weights=None):

        H, C = self.heads, self.out_channels

        if isinstance(x, Tensor):
            x: PairTensor = (x, x)

        query = self.lin_query(x[1]).view(-1, H, C)
        key = self.lin_key(x[0]).view(-1, H, C)
        value = self.lin_value(x[0]).view(-1, H, C)

        # propagate_type: (query: Tensor, key:Tensor, value: Tensor, edge_attr: OptTensor) # noqa
        out = self.propagate(edge_index, query=query, key=key, value=value, edge_attr=edge_attr, size=None)

        alpha = self._alpha
        self._alpha = None
        alpha = alpha.view(-1, self.heads, 1).mean(dim=1)
        if self.concat:
            out = out.view(-1, self.heads * self.out_channels)
        else:
            out = out.mean(dim=1)

        if self.root_weight:
            x_r = self.lin_skip(x[1])
            if self.lin_beta is not None:
                beta = self.lin_beta(torch.cat([out, x_r, out - x_r], dim=-1))
                beta = beta.sigmoid()
                out = beta * x_r + (1 - beta) * out
            else:
                out = out + x_r

        if isinstance(return_attention_weights, bool):
            assert alpha is not None
            if isinstance(edge_index, Tensor):
                return out, (edge_index, alpha)
            elif isinstance(edge_index, SparseTensor):
                return out, edge_index.set_value(alpha, layout='coo')
        else:
            return out, alpha

    def message(
        self, query_i: Tensor, query_j: Tensor, key_i: Tensor,key_j: Tensor, value_j: Tensor,
        edge_attr: OptTensor, index: Tensor, ptr: OptTensor,
        size_i: Optional[int]) -> Tensor:

        if self.lin_edge is not None:
            assert edge_attr is not None
            edge_attr = self.lin_edge(edge_attr.unsqueeze(-1)).view(-1, self.heads, self.out_channels)
            key_j = key_j + edge_attr

        alpha1 = (query_i * key_j).sum(dim=-1) / math.sqrt(self.out_channels)
        alpha1 = torch.tanh(alpha1)
        alpha2 = (query_j * key_i).sum(dim=-1) / math.sqrt(self.out_channels)
        alpha2 = torch.tanh(alpha2)
        alpha = (alpha1 + alpha2)/2
        #alpha = softmax(alpha, index, ptr, size_i)
        self._alpha = alpha
        #alpha = F.dropout(alpha, p=self.dropout, training=self.training)

        out = value_j
        if edge_attr is not None:
            #out = out + edge_attr
            out = out * edge_attr

        #out = out #* alpha.view(-1, self.heads, 1)
        return out

    def __repr__(self) -> str:
        return (f'{self.__class__.__name__}({self.in_channels}, '
                f'{self.out_channels}, heads={self.heads})')


class EdgeNetwork_sparse(torch.nn.Module):
    def __init__(self, num_linears, conv_input_dim, attn_dim, conv_output_dim, input_dim, output_dim, num_heads=4):
        super(EdgeNetwork_sparse, self).__init__()

        self.attn_dim = attn_dim
        self.attn = torch.nn.ModuleList()
        for _ in range(input_dim):
            self.attn.append(EdgeLayer_Tanh(conv_input_dim, conv_output_dim, heads=num_heads))

        self.hidden_dim = 2 * max(input_dim, output_dim)
        # self.mlp = MLP(num_linears, 2 * input_dim, self.hidden_dim, output_dim, use_bn=False, activate_func=F.elu)
        # self.multi_channel = MLP(2, input_dim * conv_output_dim, self.hidden_dim, conv_output_dim,
        #                          use_bn=False, activate_func=F.elu)
        # print("before, mlp", self.mlp)
        # print("before, multi_channel", self.multi_channel)

        hidden_dims = [self.hidden_dim] * (num_linears-1) + [output_dim]
        self.mlp = MultiLayerPerceptron(2*input_dim, hidden_dims, activation='elu')

        hidden_dims = [self.hidden_dim, conv_output_dim]
        self.multi_channel = MultiLayerPerceptron(input_dim * conv_output_dim, hidden_dims, activation='elu')
        # print("after, mlp", self.mlp)
        # print("after, multi_channel", self.multi_channel)

    def forward(self, x, edge_index, adj, flags):
        """
        :param x:  B x N x F_i
        :param adj: B x C_i x N x N
        :return: x_out: B x N x F_o, adj_out: B x C_o x N x N
        """
        mask_list = []
        x_list = []
        for _ in range(len(self.attn)):
            #_x, mask = self.attn[_](x, adj[:, _, :, :], flags)
            _x, mask = self.attn[_](x,edge_index, adj[_,:])
            mask_list.append(mask)
            x_list.append(_x)
        x_out = self.multi_channel(torch.cat(x_list, dim=-1))
        x_out = torch.tanh(x_out)
        a= torch.cat(mask_list, dim=-1)
        mlp_in = torch.cat([a, adj.permute(1,0)], dim=-1)
        # shape = mlp_in.shape
        # mlp_out = self.mlp(mlp_in.view(-1, shape[-1]))
        mlp_out = self.mlp(mlp_in)
        #_adj = mlp_out.view(shape[0], shape[1], shape[2], -1).permute(0, 3, 1, 2)
        #_adj = _adj + _adj.transpose(-1, -2)
        #adj_out = mask_adjs(_adj, flags)

        return x_out, mlp_out.permute(1,0)

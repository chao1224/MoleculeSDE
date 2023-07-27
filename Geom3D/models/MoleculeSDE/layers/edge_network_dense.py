import math
import torch

from .node_network_dense import NodeNetwork_dense, NodeNetwork_dense_03
from .common import MultiLayerPerceptron


# -------- Mask batch of node features with 0-1 flags tensor --------
def mask_x(x, flags):

    if flags is None:
        flags = torch.ones((x.shape[0], x.shape[1]), device=x.device)
    return x * flags[:,:,None]


# -------- Mask batch of adjacency matrices with 0-1 flags tensor --------
def mask_adjs(adjs, flags):
    """
    :param adjs:  B x N x N or B x C x N x N
    :param flags: B x N
    :return:
    """
    if flags is None:
        flags = torch.ones((adjs.shape[0], adjs.shape[-1]), device=adjs.device)

    if len(adjs.shape) == 4:
        flags = flags.unsqueeze(1)  # B x 1 x N
    adjs = adjs * flags.unsqueeze(-1)
    adjs = adjs * flags.unsqueeze(-2)
    return adjs


class EdgeLayer(torch.nn.Module):
    def __init__(self, in_dim, attn_dim, out_dim, num_heads, conv):
        super(EdgeLayer, self).__init__()
        self.num_heads = num_heads
        self.attn_dim = attn_dim
        self.out_dim = out_dim
        self.conv = conv

        if conv == 'GCN':
            self.func_q = NodeNetwork_dense(in_dim, attn_dim)
            self.func_k = NodeNetwork_dense(in_dim, attn_dim)
            self.func_v = NodeNetwork_dense(in_dim, out_dim)
        elif conv == 'MLP':
            hidden_dims = [2 * attn_dim, 2 * attn_dim]
            self.func_q = MultiLayerPerceptron(in_dim, hidden_dims, activation="tanh")
            self.func_k = MultiLayerPerceptron(in_dim, hidden_dims, activation="tanh")
            self.func_v = NodeNetwork_dense(in_dim, out_dim)
        else:
            raise NotImplementedError(f'{conv} not implemented.')

        self.activation = torch.tanh

    def forward(self, x, adj, flags, attention_mask=None):
        if self.conv == 'GCN':
            Q = self.func_q(x, adj)
            K = self.func_k(x, adj)
            V = self.func_v(x, adj)
        else:
            Q = self.func_q(x)
            K = self.func_k(x)
            V = self.func_v(x, adj)

        # TODO: check this
        dim_split = self.attn_dim // self.num_heads
        Q_ = torch.cat(Q.split(dim_split, 2), 0)
        K_ = torch.cat(K.split(dim_split, 2), 0)

        if attention_mask is not None:
            attention_mask = torch.cat([attention_mask for _ in range(self.num_heads)], 0)
            attention_score = Q_.bmm(K_.transpose(1, 2)) / math.sqrt(dim_split)
            A = self.activation(attention_mask + attention_score)
        else:
            A = self.activation(Q_.bmm(K_.transpose(1, 2)) / math.sqrt(dim_split))  # (B x num_heads) x N x N

        # -------- (B x num_heads) x N x N --------
        A = A.view(-1, *adj.shape)
        A = A.mean(dim=0)
        A = (A + A.transpose(-1, -2)) / 2

        return V, A


class EdgeNetwork_dense(torch.nn.Module):
    def __init__(
        self, num_linears, conv_input_dim, attn_dim, conv_output_dim, input_dim, output_dim, num_heads, conv):
        super(EdgeNetwork_dense, self).__init__()

        self.attn_dim = attn_dim
        self.attn = torch.nn.ModuleList()
        for _ in range(input_dim):
            self.attn.append(EdgeLayer(conv_input_dim, self.attn_dim, conv_output_dim, num_heads=num_heads, conv=conv))

        self.hidden_dim = 2 * max(input_dim, output_dim)

        hidden_dims = [self.hidden_dim] * (num_linears-1) + [output_dim]
        self.mlp = MultiLayerPerceptron(2*input_dim, hidden_dims, activation='elu')

        hidden_dims = [self.hidden_dim, conv_output_dim]
        self.multi_channel = MultiLayerPerceptron(input_dim * conv_output_dim, hidden_dims, activation='elu')
        
        self.activation = torch.tanh

    def forward(self, x, adj, flags):
        """
        :param x:  B x N x F_i
        :param adj: B x C_i x N x N
        :return: x_out: B x N x F_o, adj_out: B x C_o x N x N
        """
        mask_list = []
        x_list = []
        for _ in range(len(self.attn)):
            _x, mask = self.attn[_](x, adj[:, _, :, :], flags)
            mask_list.append(mask.unsqueeze(-1))
            x_list.append(_x)
        x_out = mask_x(self.multi_channel(torch.cat(x_list, dim=-1)), flags)
        x_out = self.activation(x_out)

        mlp_in = torch.cat([torch.cat(mask_list, dim=-1), adj.permute(0, 2, 3, 1)], dim=-1)
        shape = mlp_in.shape
        
        mlp_out = self.mlp(mlp_in.view(-1, shape[-1]))
        _adj = mlp_out.view(shape[0], shape[1], shape[2], -1).permute(0, 3, 1, 2)
        _adj = _adj + _adj.transpose(-1, -2)
        adj_out = mask_adjs(_adj, flags)

        return x_out, adj_out


class EdgeLayer_03(torch.nn.Module):
    def __init__(self, in_dim, attn_dim, out_dim, num_heads, conv, node_3D_dim):
        super(EdgeLayer_03, self).__init__()
        self.num_heads = num_heads
        self.attn_dim = attn_dim
        self.out_dim = out_dim
        self.conv = conv

        if conv == 'GCN':
            self.func_q = NodeNetwork_dense_03(in_dim+node_3D_dim, attn_dim)
            self.func_k = NodeNetwork_dense_03(in_dim+node_3D_dim, attn_dim)
            self.func_v = NodeNetwork_dense_03(in_dim+node_3D_dim, out_dim)
        elif conv == 'MLP':
            hidden_dims = [2 * attn_dim, 2 * attn_dim]
            self.func_q = MultiLayerPerceptron(in_dim+node_3D_dim, hidden_dims, activation="tanh")
            self.func_k = MultiLayerPerceptron(in_dim+node_3D_dim, hidden_dims, activation="tanh")
            self.func_v = NodeNetwork_dense_03(in_dim+node_3D_dim, out_dim)
        else:
            raise NotImplementedError(f'{conv} not implemented.')

        self.activation = torch.tanh

    def forward(self, x, adj, node_3D_repr, flags, attention_mask=None):
        if self.conv == 'GCN':
            Q = self.func_q(x, adj, node_3D_repr)
            K = self.func_k(x, adj, node_3D_repr)
            V = self.func_v(x, adj, node_3D_repr)
        else:
            Q = self.func_q(torch.cat((x, node_3D_repr),-1))
            K = self.func_k(torch.cat((x, node_3D_repr),-1))
            V = self.func_v(x, adj, node_3D_repr)

        # TODO: check this
        dim_split = self.attn_dim // self.num_heads
        Q_ = torch.cat(Q.split(dim_split, 2), 0)
        K_ = torch.cat(K.split(dim_split, 2), 0)

        if attention_mask is not None:
            attention_mask = torch.cat([attention_mask for _ in range(self.num_heads)], 0)
            attention_score = Q_.bmm(K_.transpose(1, 2)) / math.sqrt(dim_split)
            A = self.activation(attention_mask + attention_score)
        else:
            A = self.activation(Q_.bmm(K_.transpose(1, 2)) / math.sqrt(dim_split))  # (B x num_heads) x N x N

        # -------- (B x num_heads) x N x N --------
        A = A.view(-1, *adj.shape)
        A = A.mean(dim=0)
        A = (A + A.transpose(-1, -2)) / 2

        return V, A


class EdgeNetwork_dense_03(torch.nn.Module):
    def __init__(
        self, num_linears, conv_input_dim, attn_dim, conv_output_dim, input_dim, output_dim, num_heads, conv, node_3D_dim):
        super(EdgeNetwork_dense_03, self).__init__()

        self.attn_dim = attn_dim
        self.attn = torch.nn.ModuleList()
        for _ in range(input_dim):
            self.attn.append(EdgeLayer_03(conv_input_dim, self.attn_dim, conv_output_dim, num_heads=num_heads, conv=conv, node_3D_dim=node_3D_dim))

        self.hidden_dim = 2 * max(input_dim, output_dim)

        hidden_dims = [self.hidden_dim] * (num_linears-1) + [output_dim]
        self.mlp = MultiLayerPerceptron(2*input_dim, hidden_dims, activation='elu')

        hidden_dims = [self.hidden_dim, conv_output_dim]
        self.multi_channel = MultiLayerPerceptron(input_dim * conv_output_dim, hidden_dims, activation='elu')
        
        self.activation = torch.tanh

    def forward(self, x, adj, node_3D_repr, flags):
        """
        :param x:  B x N x F_i
        :param adj: B x C_i x N x N
        :return: x_out: B x N x F_o, adj_out: B x C_o x N x N
        """
        mask_list = []
        x_list = []
        for _ in range(len(self.attn)):
            _x, mask = self.attn[_](x, adj[:, _, :, :], node_3D_repr, flags)
            mask_list.append(mask.unsqueeze(-1))
            x_list.append(_x)
        x_out = mask_x(self.multi_channel(torch.cat(x_list, dim=-1)), flags)
        x_out = self.activation(x_out)

        mlp_in = torch.cat([torch.cat(mask_list, dim=-1), adj.permute(0, 2, 3, 1)], dim=-1)
        shape = mlp_in.shape
        
        mlp_out = self.mlp(mlp_in.view(-1, shape[-1]))
        _adj = mlp_out.view(shape[0], shape[1], shape[2], -1).permute(0, 3, 1, 2)
        _adj = _adj + _adj.transpose(-1, -2)
        adj_out = mask_adjs(_adj, flags)

        return x_out, adj_out

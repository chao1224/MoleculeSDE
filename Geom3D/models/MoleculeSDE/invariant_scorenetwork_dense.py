import torch
from .layers import MultiLayerPerceptron, EdgeNetwork_dense, NodeNetwork_dense, EdgeNetwork_dense_03, NodeNetwork_dense_03


def mask_x(x, flags):

    if flags is None:
        flags = torch.ones((x.shape[0], x.shape[1]), device=x.device)
    return x * flags[:,:,None]


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


def pow_tensor(x, cnum=1):
    # x : B x N x N
    x_ = x.clone()
    xc = [x.unsqueeze(1)]
    for _ in range(cnum-1):
        x_ = torch.bmm(x_, x)
        xc.append(x_.unsqueeze(1))
    xc = torch.cat(xc, dim=1)

    return xc


class EdgeScoreNetwork_dense(torch.nn.Module):
    def __init__(
        self, dim3D, nhid, num_layers, num_linears, c_init, c_hid, c_final, adim, num_heads, conv
    ):
        super(EdgeScoreNetwork_dense, self).__init__()

        self.adim = adim
        self.num_heads = num_heads
        self.conv = conv
        self.c_init = c_init
        self.c_hid = c_hid
        self.c_final = c_final
        self.adim = adim
        self.num_layers = num_layers
        self.num_linears = num_linears
        self.nfeat = dim3D
        self.nhid = nhid

        self.layers = torch.nn.ModuleList()
        for _ in range(self.num_layers):
            if _ == 0:
                self.layers.append(
                    EdgeNetwork_dense(self.num_linears, self.nfeat, self.nhid, self.nhid, self.c_init, self.c_hid, self.num_heads, self.conv))
            elif _ == self.num_layers - 1:
                self.layers.append(
                    EdgeNetwork_dense(self.num_linears, self.nhid, self.adim, self.nhid, self.c_hid, self.c_final, self.num_heads, self.conv))
            else:
                self.layers.append(
                    EdgeNetwork_dense(self.num_linears, self.nhid, self.adim, self.nhid, self.c_hid, self.c_hid, self.num_heads, self.conv))

        self.fdim = self.c_hid * (self.num_layers - 1) + self.c_final + self.c_init
        hidden_dims = [2*self.fdim, 2*self.fdim, 1]
        self.final = MultiLayerPerceptron(input_dim=self.fdim, hidden_dims=hidden_dims, activation="silu")

    def forward(self, x, adj, flags):
        adjc = pow_tensor(adj, self.c_init)

        adj_list = [adjc]
        for layer_idx in range(self.num_layers):
            x, adjc = self.layers[layer_idx](x, adjc, flags)
            adj_list.append(adjc)

        adjs = torch.cat(adj_list, dim=1).permute(0, 2, 3, 1)
        out_shape = adjs.shape[:-1]  # B x N x N
        score = self.final(adjs).view(*out_shape)
        
        max_node_num = adjs.size()[1]
        mask = torch.ones([max_node_num, max_node_num]) - torch.eye(max_node_num)  # mask out the diagonal
        mask.unsqueeze_(0)
        mask = mask.to(score.device)
        score = score * mask
        score = mask_adjs(score, flags)

        return score


class NodeScoreNetwork_dense(torch.nn.Module):
    def __init__(self, nfeat, depth, nhid, nout):
        super(NodeScoreNetwork_dense, self).__init__()

        self.nfeat = nfeat
        self.depth = depth
        self.nhid = nhid
        self.nout = nout

        self.layers = torch.nn.ModuleList()
        for _ in range(self.depth):
            if _ == 0:
                self.layers.append(NodeNetwork_dense(self.nfeat, self.nhid))
            else:
                self.layers.append(NodeNetwork_dense(self.nhid, self.nhid))

        self.fdim = self.nfeat + self.depth * self.nhid
        hidden_dims = [2*self.fdim, 2*self.fdim, self.nout]
        self.final = MultiLayerPerceptron(input_dim=self.fdim, hidden_dims=hidden_dims, activation="silu")

        self.activation = torch.tanh

    def forward(self, x, adj, flags):
        x_list = [x]
        for _ in range(self.depth):
            x = self.layers[_](x, adj)
            x = self.activation(x)
            x_list.append(x)

        xs = torch.cat(x_list, dim=-1) # B x N x (F + num_layers x H)
        out_shape = (adj.shape[0], adj.shape[1], -1)
        x = self.final(xs).view(*out_shape)

        x = mask_x(x, flags)

        return x


class EdgeScoreNetwork_dense_03(torch.nn.Module):
    def __init__(
        self, dim3D, nhid, num_layers, num_linears, c_init, c_hid, c_final, adim, num_heads, conv
    ):
        super(EdgeScoreNetwork_dense_03, self).__init__()

        self.adim = adim
        self.num_heads = num_heads
        self.conv = conv
        self.c_init = c_init
        self.c_hid = c_hid
        self.c_final = c_final
        self.adim = adim
        self.num_layers = num_layers
        self.num_linears = num_linears
        self.nfeat = dim3D
        self.nhid = nhid

        self.layers = torch.nn.ModuleList()
        for _ in range(self.num_layers):
            if _ == 0:
                self.layers.append(
                    EdgeNetwork_dense_03(self.num_linears, self.nfeat, self.nhid, self.nhid, self.c_init, self.c_hid, self.num_heads, self.conv, dim3D//2))
            elif _ == self.num_layers - 1:
                self.layers.append(
                    EdgeNetwork_dense_03(self.num_linears, self.nhid, self.adim, self.nhid, self.c_hid, self.c_final, self.num_heads, self.conv, dim3D//2))
            else:
                self.layers.append(
                    EdgeNetwork_dense_03(self.num_linears, self.nhid, self.adim, self.nhid, self.c_hid, self.c_hid, self.num_heads, self.conv, dim3D//2))

        self.fdim = self.c_hid * (self.num_layers - 1) + self.c_final + self.c_init
        hidden_dims = [2*self.fdim, 2*self.fdim, 1]
        self.final = MultiLayerPerceptron(input_dim=self.fdim, hidden_dims=hidden_dims, activation="silu")

    def forward(self, x, adj, node_3D_repr, flags):
        adjc = pow_tensor(adj, self.c_init)

        adj_list = [adjc]
        for layer_idx in range(self.num_layers):
            x, adjc = self.layers[layer_idx](x, adjc, node_3D_repr, flags)
            adj_list.append(adjc)

        adjs = torch.cat(adj_list, dim=1).permute(0, 2, 3, 1)
        out_shape = adjs.shape[:-1]  # B x N x N
        score = self.final(adjs).view(*out_shape)
        
        max_node_num = adjs.size()[1]
        mask = torch.ones([max_node_num, max_node_num]) - torch.eye(max_node_num)  # mask out the diagonal
        mask.unsqueeze_(0)
        mask = mask.to(score.device)
        score = score * mask
        score = mask_adjs(score, flags)

        return score


class NodeScoreNetwork_dense_03(torch.nn.Module):
    def __init__(self, nfeat, depth, nhid, nout):
        super(NodeScoreNetwork_dense_03, self).__init__()

        self.nfeat = nfeat
        self.depth = depth
        self.nhid = nhid
        self.nout = nout

        self.layers = torch.nn.ModuleList()
        for _ in range(self.depth):
            if _ == 0:
                self.layers.append(NodeNetwork_dense_03(self.nfeat + self.nfeat//2, self.nhid))
            else:
                self.layers.append(NodeNetwork_dense_03(self.nhid + self.nfeat//2, self.nhid))

        self.fdim = self.nfeat + self.depth * self.nhid
        hidden_dims = [2*self.fdim, 2*self.fdim, self.nout]
        self.final = MultiLayerPerceptron(input_dim=self.fdim, hidden_dims=hidden_dims, activation="silu")

        self.activation = torch.tanh

    def forward(self, x, adj, node_3D_repr, flags):
        x_list = [x]
        for _ in range(self.depth):
            x = self.layers[_](x, adj, node_3D_repr)
            x = self.activation(x)
            x_list.append(x)

        xs = torch.cat(x_list, dim=-1) # B x N x (F + num_layers x H)
        out_shape = (adj.shape[0], adj.shape[1], -1)
        x = self.final(xs).view(*out_shape)

        x = mask_x(x, flags)

        return x
import torch
import torch.nn.functional as F
from torch import nn
import numpy as np
from torch_scatter import scatter
from torch_geometric.utils import to_dense_adj, to_dense_batch
from .invariant_scorenetwork_dense import EdgeScoreNetwork_dense, NodeScoreNetwork_dense, EdgeScoreNetwork_dense_03, NodeScoreNetwork_dense_03
from .SDE_dense import VPSDE, VESDE, subVPSDE

EPSILON = 1e-6


class SDEModel3Dto2D_node_adj_dense(torch.nn.Module):
    def __init__(
        self, dim3D, nhid, num_layers, num_linears, c_hid, c_final, adim, emb_dim,
        beta_min, beta_max, num_diffusion_timesteps, c_init=1, num_heads=4, conv='MLP', noise_mode="discrete", SDE_type='VE', num_class_X=119, noise_on_one_hot=True):
        super(SDEModel3Dto2D_node_adj_dense, self).__init__()

        self.emb_dim = emb_dim

        self.beta_min = beta_min
        self.beta_max = beta_max
        self.num_diffusion_timesteps = num_diffusion_timesteps
        self.nfeat = dim3D
        self.nhid = nhid
        self.num_layers = num_layers
        self.num_linears = num_linears
        self.c_init = c_init
        self.c_hid = c_hid
        self.c_final = c_final
        self.adim = adim
        self.num_heads = num_heads
        self.conv = conv
        self.noise_mode = noise_mode
        self.SDE_type = SDE_type

        if self.SDE_type == "VE":
            self.sde_x = VESDE(sigma_min=self.beta_min, sigma_max=self.beta_max, N=self.num_diffusion_timesteps)
            self.sde_adj = VESDE(sigma_min=self.beta_min, sigma_max=self.beta_max, N=self.num_diffusion_timesteps)
        elif self.SDE_type == "VP":
            self.sde_x = VPSDE(beta_min=self.beta_min, beta_max=self.beta_max, N=self.num_diffusion_timesteps)
            self.sde_adj = VPSDE(beta_min=self.beta_min, beta_max=self.beta_max, N=self.num_diffusion_timesteps)

        self.num_class_X = num_class_X
        self.noise_on_one_hot = noise_on_one_hot
        if self.noise_on_one_hot:
            self.embedding_X = nn.Linear(self.num_class_X, self.nfeat)
        else:
            self.embedding_X = nn.Linear(1, self.nfeat)

        self.embedding_3D = nn.Linear(self.nfeat, self.nfeat)

        self.edge_score_network = EdgeScoreNetwork_dense(
            dim3D=self.nfeat, nhid=self.nhid,
            num_layers=self.num_layers,
            num_linears=self.num_linears,
            c_init=self.c_init, c_hid=self.c_hid, c_final=self.c_final, adim=self.adim,
            num_heads=4, conv=self.conv)

        if noise_on_one_hot:
            nout = num_class_X
        else:
            nout = 1
        self.node_score_network = NodeScoreNetwork_dense(nfeat=self.nfeat, depth=self.num_layers, nhid=self.nhid, nout=nout)

        return

    def get_score_fn(self, sde, model, train=True, continuous=True):

        if not train:
            model.eval()
        model_fn = model

        if isinstance(sde, VPSDE) or isinstance(sde, subVPSDE):
            def score_fn(x, adj, flags, t):
                # Scale neural network output by standard deviation and flip sign
                if continuous:
                    # given 3d representation and current adj, return current score function
                    score = model_fn(x, adj, flags)
                    std = sde.marGINal_prob(torch.zeros_like(adj), t)[1]
                else:
                    raise NotImplementedError(f"Discrete not supported")
                score = -score / std[:, None, None]
                return score

        elif isinstance(sde, VESDE):
            def score_fn(x, adj, flags, t):
                if continuous:
                    score = model_fn(x, adj, flags)
                    std = sde.marGINal_prob(torch.zeros_like(adj), t)[1]
                else:
                    raise NotImplementedError(f"Discrete not supported")
                score = -score / std[:, None, None]
                return score

        else:
            raise NotImplementedError(f"SDE class {sde.__class__.__name__} not supported.")

        return score_fn

    def forward(self, node_3D_repr, data, continuous, train, reduce_mean, anneal_power):
        """
        Args:
            node_3D_repr: 3D node representation (num_node, hidden)
        Output:
            gradient (score)
        """
        device = node_3D_repr.device
        node2graph = data.batch

        if self.noise_mode == "discrete":
            t = torch.randint(0, self.num_diffusion_timesteps, size=(data.num_graphs // 2 + 1,), device=device)
            t = torch.cat([t, self.num_diffusion_timesteps - t - 1], dim=0)[:data.num_graphs]
            t = t / self.num_diffusion_timesteps * (1 - EPSILON) + EPSILON  # normalize to [0, 1]
        else:
            t = torch.rand(data.num_graphs, device=device) * (1 - EPSILON) + EPSILON

        # To extract bond type
        # https://github.com/snap-stanford/ogb/blob/master/ogb/utils/features.py#L97
        # TODO: double-check + 1
        edge_attr = data.edge_attr[:, 0].float() + 1

        # calculate max_num_nodes
        batch_size = node2graph.max().item() + 1
        one = node2graph.new_ones(node2graph.size(0))
        num_nodes = scatter(one, node2graph, dim=0, dim_size=batch_size, reduce='add')
        max_num_nodes = num_nodes.max().item()
        
        adj = to_dense_adj(data.edge_index, node2graph, edge_attr=edge_attr, max_num_nodes=max_num_nodes)
        node_3D_repr, _ = to_dense_batch(node_3D_repr, node2graph, max_num_nodes=max_num_nodes)  # [B, max_num_nodes, hdim]
        z, _ = to_dense_batch(data.x[:, 0], node2graph, max_num_nodes=max_num_nodes)  # [B, max_num_nodes, 1]

        # pertube adj
        flags = node_flags(adj)
        z_adj = gen_noise(adj, flags, sym=True)
        mean_adj, std_adj = self.sde_adj.marGINal_prob(adj, t)
        perturbed_adj = mean_adj + std_adj[:, None, None] * z_adj
        perturbed_adj = mask_adjs(perturbed_adj, flags)
        score_fn_adj = self.get_score_fn(self.sde_adj, self.edge_score_network, train=train, continuous=continuous)

        # pertube x
        if self.noise_on_one_hot:
            z_one_hot = F.one_hot(z, self.num_class_X).float()  # [B, max_num_nodes, num_class_X]
            z_x = gen_noise(z_one_hot, flags, sym=False)  # [B, max_num_nodes, num_class_X]
            mean_x, std_x = self.sde_x.marGINal_prob(z_one_hot, t)  # [B, max_num_nodes, num_class_X], [B]
            perturbed_x = mean_x + std_x[:, None, None] * z_x  # [B, max_num_nodes, num_class_X]
        else:
            z = z.float().unsqueeze(2)
            z_x = gen_noise(z, flags, sym=False)  # [B, max_num_nodes, 1]
            mean_x, std_x = self.sde_x.marGINal_prob(z, t)  # [B, max_num_nodes, 1], [B]
            perturbed_x = mean_x + std_x[:, None, None] * z_x  # [B, max_num_nodes, 1]
        perturbed_x = mask_x(perturbed_x, flags)  # [B, max_num_nodes, num_class_X] or [B, max_num_nodes, 1]
        score_fn_x = self.get_score_fn(self.sde_x, self.node_score_network, train=train, continuous=continuous)  # function with output dim [B, max_num_nodes, num_class_X] or [B, max_num_nodes, 1]
        
        # 3Dx + pertubed atomx
        perturbed_x = self.embedding_3D(node_3D_repr) + self.embedding_X(perturbed_x)  # [B, max_num_nodes, hdim]
        score_adj = score_fn_adj(perturbed_x, perturbed_adj, flags, t)  # [B, max_num_nodes, max_num_nodes]
        score_x = score_fn_x(perturbed_x, perturbed_adj, flags, t)  # [B, max_num_nodes, num_class_X] or [B, max_num_nodes, 1]

        reduce_op = torch.mean if reduce_mean else lambda *args, **kwargs: 0.5 * torch.sum(*args, **kwargs)

        # TODO: we need to double-check this
        if anneal_power == 0:
            losses_x = torch.square(score_x + z_x)  # [B, max_num_nodes, num_class_X] or [B, max_num_nodes, 1]
            losses_adj = torch.square(score_adj + z_adj)  # [B, max_num_nodes, max_num_nodes]

        else:
            annealed_std_x = std_x ** anneal_power  # [B]
            annealed_std_x = annealed_std_x.unsqueeze(1).unsqueeze(2)  # [B, 1, 1]
            losses_x = torch.square(score_x + z_x) * annealed_std_x  # [B, max_num_nodes, num_class_X] or [B, max_num_nodes, 1]
            
            annealed_std_adj = std_adj ** anneal_power  # [B]
            annealed_std_adj = annealed_std_adj.unsqueeze(1).unsqueeze(2)  # [B, 1, 1]
            losses_adj = torch.square(score_adj + z_adj) * annealed_std_adj  # [B, max_num_nodes, max_num_nodes]

        losses_x = reduce_op(losses_x.reshape(losses_x.shape[0], -1), dim=-1)
        losses_adj = reduce_op(losses_adj.reshape(losses_adj.shape[0], -1), dim=-1)

        return torch.mean(losses_x), torch.mean(losses_adj)


class SDEModel3Dto2D_node_adj_dense_02(torch.nn.Module):
    def __init__(
        self, dim3D, nhid, num_layers, num_linears, c_hid, c_final, adim, emb_dim,
        beta_min, beta_max, num_diffusion_timesteps, c_init=1, num_heads=4, conv='MLP', noise_mode="discrete", SDE_type='VE', num_class_X=119, noise_on_one_hot=True):
        super(SDEModel3Dto2D_node_adj_dense_02, self).__init__()

        self.emb_dim = emb_dim

        self.beta_min = beta_min
        self.beta_max = beta_max
        self.num_diffusion_timesteps = num_diffusion_timesteps
        self.nfeat = dim3D
        self.nhid = nhid
        self.num_layers = num_layers
        self.num_linears = num_linears
        self.c_init = c_init
        self.c_hid = c_hid
        self.c_final = c_final
        self.adim = adim
        self.num_heads = num_heads
        self.conv = conv
        self.noise_mode = noise_mode
        self.SDE_type = SDE_type

        if self.SDE_type == "VE":
            self.sde_x = VESDE(sigma_min=self.beta_min, sigma_max=self.beta_max, N=self.num_diffusion_timesteps)
            self.sde_adj = VESDE(sigma_min=self.beta_min, sigma_max=self.beta_max, N=self.num_diffusion_timesteps)
        elif self.SDE_type == "VP":
            self.sde_x = VPSDE(beta_min=self.beta_min, beta_max=self.beta_max, N=self.num_diffusion_timesteps)
            self.sde_adj = VPSDE(beta_min=self.beta_min, beta_max=self.beta_max, N=self.num_diffusion_timesteps)

        self.num_class_X = num_class_X
        self.noise_on_one_hot = noise_on_one_hot
        if self.noise_on_one_hot:
            self.embedding_X = nn.Linear(self.num_class_X, self.nfeat)
        else:
            self.embedding_X = nn.Linear(1, self.nfeat)

        self.embedding_3D = nn.Linear(self.nfeat, self.nfeat)

        self.edge_score_network = EdgeScoreNetwork_dense(
            dim3D=2*self.nfeat, nhid=self.nhid,
            num_layers=self.num_layers,
            num_linears=self.num_linears,
            c_init=self.c_init, c_hid=self.c_hid, c_final=self.c_final, adim=self.adim,
            num_heads=4, conv=self.conv)

        if noise_on_one_hot:
            nout = num_class_X
        else:
            nout = 1
        self.node_score_network = NodeScoreNetwork_dense(nfeat=2*self.nfeat, depth=self.num_layers, nhid=self.nhid, nout=nout)

        return

    def get_score_fn(self, sde, model, train=True, continuous=True):

        if not train:
            model.eval()
        model_fn = model

        if isinstance(sde, VPSDE) or isinstance(sde, subVPSDE):
            def score_fn(x, adj, flags, t):
                # Scale neural network output by standard deviation and flip sign
                if continuous:
                    # given 3d representation and current adj, return current score function
                    score = model_fn(x, adj, flags)
                    std = sde.marGINal_prob(torch.zeros_like(adj), t)[1]
                else:
                    raise NotImplementedError(f"Discrete not supported")
                score = -score / std[:, None, None]
                return score

        elif isinstance(sde, VESDE):
            def score_fn(x, adj, flags, t):
                if continuous:
                    score = model_fn(x, adj, flags)
                    std = sde.marGINal_prob(torch.zeros_like(adj), t)[1]
                else:
                    raise NotImplementedError(f"Discrete not supported")
                score = -score / std[:, None, None]
                return score

        else:
            raise NotImplementedError(f"SDE class {sde.__class__.__name__} not supported.")

        return score_fn

    def forward(self, node_3D_repr, data, continuous, train, reduce_mean, anneal_power):
        """
        Args:
            node_3D_repr: 3D node representation (num_node, hidden)
        Output:
            gradient (score)
        """
        device = node_3D_repr.device
        node2graph = data.batch

        if self.noise_mode == "discrete":
            t = torch.randint(0, self.num_diffusion_timesteps, size=(data.num_graphs // 2 + 1,), device=device)
            t = torch.cat([t, self.num_diffusion_timesteps - t - 1], dim=0)[:data.num_graphs]
            t = t / self.num_diffusion_timesteps * (1 - EPSILON) + EPSILON  # normalize to [0, 1]
        else:
            t = torch.rand(data.num_graphs, device=device) * (1 - EPSILON) + EPSILON

        # To extract bond type
        # https://github.com/snap-stanford/ogb/blob/master/ogb/utils/features.py#L97
        # TODO: double-check + 1
        edge_attr = data.edge_attr[:, 0].float() + 1

        # calculate max_num_nodes
        batch_size = node2graph.max().item() + 1
        one = node2graph.new_ones(node2graph.size(0))
        num_nodes = scatter(one, node2graph, dim=0, dim_size=batch_size, reduce='add')
        max_num_nodes = num_nodes.max().item()
        
        adj = to_dense_adj(data.edge_index, node2graph, edge_attr=edge_attr, max_num_nodes=max_num_nodes)
        node_3D_repr, _ = to_dense_batch(node_3D_repr, node2graph, max_num_nodes=max_num_nodes)  # [B, max_num_nodes, hdim]
        z, _ = to_dense_batch(data.x[:, 0], node2graph, max_num_nodes=max_num_nodes)  # [B, max_num_nodes, 1]

        # pertube adj
        flags = node_flags(adj)
        z_adj = gen_noise(adj, flags, sym=True)
        mean_adj, std_adj = self.sde_adj.marGINal_prob(adj, t)
        perturbed_adj = mean_adj + std_adj[:, None, None] * z_adj
        perturbed_adj = mask_adjs(perturbed_adj, flags)
        score_fn_adj = self.get_score_fn(self.sde_adj, self.edge_score_network, train=train, continuous=continuous)

        # pertube x
        if self.noise_on_one_hot:
            z_one_hot = F.one_hot(z, self.num_class_X).float()  # [B, max_num_nodes, num_class_X]
            z_x = gen_noise(z_one_hot, flags, sym=False)  # [B, max_num_nodes, num_class_X]
            mean_x, std_x = self.sde_x.marGINal_prob(z_one_hot, t)  # [B, max_num_nodes, num_class_X], [B]
            perturbed_x = mean_x + std_x[:, None, None] * z_x  # [B, max_num_nodes, num_class_X]
        else:
            z = z.float().unsqueeze(2)
            z_x = gen_noise(z, flags, sym=False)  # [B, max_num_nodes, 1]
            mean_x, std_x = self.sde_x.marGINal_prob(z, t)  # [B, max_num_nodes, 1], [B]
            perturbed_x = mean_x + std_x[:, None, None] * z_x  # [B, max_num_nodes, 1]
        perturbed_x = mask_x(perturbed_x, flags)  # [B, max_num_nodes, num_class_X] or [B, max_num_nodes, 1]
        score_fn_x = self.get_score_fn(self.sde_x, self.node_score_network, train=train, continuous=continuous)  # function with output dim [B, max_num_nodes, num_class_X] or [B, max_num_nodes, 1]
        
        # 3Dx + pertubed atomx
        #perturbed_x = self.embedding_3D(node_3D_repr) + self.embedding_X(perturbed_x)  # [B, max_num_nodes, hdim]
        perturbed_x = torch.cat([self.embedding_3D(node_3D_repr), self.embedding_X(perturbed_x)],-1)  # [B, max_num_nodes, 2*hdim]
        score_adj = score_fn_adj(perturbed_x, perturbed_adj, flags, t)  # [B, max_num_nodes, max_num_nodes]
        score_x = score_fn_x(perturbed_x, perturbed_adj, flags, t)  # [B, max_num_nodes, num_class_X] or [B, max_num_nodes, 1]

        reduce_op = torch.mean if reduce_mean else lambda *args, **kwargs: 0.5 * torch.sum(*args, **kwargs)

        # TODO: we need to double-check this
        if anneal_power == 0:
            losses_x = torch.square(score_x + z_x)  # [B, max_num_nodes, num_class_X] or [B, max_num_nodes, 1]
            losses_adj = torch.square(score_adj + z_adj)  # [B, max_num_nodes, max_num_nodes]

        else:
            annealed_std_x = std_x ** anneal_power  # [B]
            annealed_std_x = annealed_std_x.unsqueeze(1).unsqueeze(2)  # [B, 1, 1]
            losses_x = torch.square(score_x + z_x) * annealed_std_x  # [B, max_num_nodes, num_class_X] or [B, max_num_nodes, 1]
            
            annealed_std_adj = std_adj ** anneal_power  # [B]
            annealed_std_adj = annealed_std_adj.unsqueeze(1).unsqueeze(2)  # [B, 1, 1]
            losses_adj = torch.square(score_adj + z_adj) * annealed_std_adj  # [B, max_num_nodes, max_num_nodes]

        losses_x = reduce_op(losses_x.reshape(losses_x.shape[0], -1), dim=-1)
        losses_adj = reduce_op(losses_adj.reshape(losses_adj.shape[0], -1), dim=-1)

        return torch.mean(losses_x), torch.mean(losses_adj)


class SDEModel3Dto2D_node_adj_dense_03(torch.nn.Module):
    def __init__(
        self, dim3D, nhid, num_layers, num_linears, c_hid, c_final, adim, emb_dim,
        beta_min, beta_max, num_diffusion_timesteps, c_init=1, num_heads=4, conv='MLP', noise_mode="discrete", SDE_type='VE', num_class_X=119, noise_on_one_hot=True):
        super(SDEModel3Dto2D_node_adj_dense_03, self).__init__()

        self.emb_dim = emb_dim

        self.beta_min = beta_min
        self.beta_max = beta_max
        self.num_diffusion_timesteps = num_diffusion_timesteps
        self.nfeat = dim3D
        self.nhid = nhid
        self.num_layers = num_layers
        self.num_linears = num_linears
        self.c_init = c_init
        self.c_hid = c_hid
        self.c_final = c_final
        self.adim = adim
        self.num_heads = num_heads
        self.conv = conv
        self.noise_mode = noise_mode
        self.SDE_type = SDE_type

        if self.SDE_type == "VE":
            self.sde_x = VESDE(sigma_min=self.beta_min, sigma_max=self.beta_max, N=self.num_diffusion_timesteps)
            self.sde_adj = VESDE(sigma_min=self.beta_min, sigma_max=self.beta_max, N=self.num_diffusion_timesteps)
        elif self.SDE_type == "VP":
            self.sde_x = VPSDE(beta_min=self.beta_min, beta_max=self.beta_max, N=self.num_diffusion_timesteps)
            self.sde_adj = VPSDE(beta_min=self.beta_min, beta_max=self.beta_max, N=self.num_diffusion_timesteps)

        self.num_class_X = num_class_X
        self.noise_on_one_hot = noise_on_one_hot
        if self.noise_on_one_hot:
            self.embedding_X = nn.Linear(self.num_class_X, self.nfeat)
        else:
            self.embedding_X = nn.Linear(1, self.nfeat)

        self.embedding_3D = nn.Linear(self.nfeat, self.nfeat)

        self.edge_score_network = EdgeScoreNetwork_dense_03(
            dim3D=2*self.nfeat, nhid=self.nhid,
            num_layers=self.num_layers,
            num_linears=self.num_linears,
            c_init=self.c_init, c_hid=self.c_hid, c_final=self.c_final, adim=self.adim,
            num_heads=4, conv=self.conv)

        if noise_on_one_hot:
            nout = num_class_X
        else:
            nout = 1
        self.node_score_network = NodeScoreNetwork_dense_03(nfeat=2*self.nfeat, depth=self.num_layers, nhid=self.nhid, nout=nout)

        return

    def get_score_fn(self, sde, model, train=True, continuous=True):

        if not train:
            model.eval()
        model_fn = model

        if isinstance(sde, VPSDE) or isinstance(sde, subVPSDE):
            def score_fn(x, adj, node_3D_repr, flags, t):
                # Scale neural network output by standard deviation and flip sign
                if continuous:
                    # given 3d representation and current adj, return current score function
                    score = model_fn(x, adj, node_3D_repr, flags)
                    std = sde.marGINal_prob(torch.zeros_like(adj), t)[1]
                else:
                    raise NotImplementedError(f"Discrete not supported")
                score = -score / std[:, None, None]
                return score

        elif isinstance(sde, VESDE):
            def score_fn(x, adj, node_3D_repr, flags, t):
                if continuous:
                    score = model_fn(x, adj, node_3D_repr, flags)
                    std = sde.marGINal_prob(torch.zeros_like(adj), t)[1]
                else:
                    raise NotImplementedError(f"Discrete not supported")
                score = -score / std[:, None, None]
                return score

        else:
            raise NotImplementedError(f"SDE class {sde.__class__.__name__} not supported.")

        return score_fn

    def forward(self, node_3D_repr, data, continuous, train, reduce_mean, anneal_power):
        """
        Args:
            node_3D_repr: 3D node representation (num_node, hidden)
        Output:
            gradient (score)
        """
        device = node_3D_repr.device
        node2graph = data.batch

        if self.noise_mode == "discrete":
            t = torch.randint(0, self.num_diffusion_timesteps, size=(data.num_graphs // 2 + 1,), device=device)
            t = torch.cat([t, self.num_diffusion_timesteps - t - 1], dim=0)[:data.num_graphs]
            t = t / self.num_diffusion_timesteps * (1 - EPSILON) + EPSILON  # normalize to [0, 1]
        else:
            t = torch.rand(data.num_graphs, device=device) * (1 - EPSILON) + EPSILON

        # To extract bond type
        # https://github.com/snap-stanford/ogb/blob/master/ogb/utils/features.py#L97
        # TODO: double-check + 1
        edge_attr = data.edge_attr[:, 0].float() + 1

        # calculate max_num_nodes
        batch_size = node2graph.max().item() + 1
        one = node2graph.new_ones(node2graph.size(0))
        num_nodes = scatter(one, node2graph, dim=0, dim_size=batch_size, reduce='add')
        max_num_nodes = num_nodes.max().item()
        
        adj = to_dense_adj(data.edge_index, node2graph, edge_attr=edge_attr, max_num_nodes=max_num_nodes)
        node_3D_repr, _ = to_dense_batch(node_3D_repr, node2graph, max_num_nodes=max_num_nodes)  # [B, max_num_nodes, hdim]
        z, _ = to_dense_batch(data.x[:, 0], node2graph, max_num_nodes=max_num_nodes)  # [B, max_num_nodes, 1]

        # pertube adj
        flags = node_flags(adj)
        z_adj = gen_noise(adj, flags, sym=True)
        mean_adj, std_adj = self.sde_adj.marGINal_prob(adj, t)
        perturbed_adj = mean_adj + std_adj[:, None, None] * z_adj
        perturbed_adj = mask_adjs(perturbed_adj, flags)
        score_fn_adj = self.get_score_fn(self.sde_adj, self.edge_score_network, train=train, continuous=continuous)

        # pertube x
        if self.noise_on_one_hot:
            z_one_hot = F.one_hot(z, self.num_class_X).float()  # [B, max_num_nodes, num_class_X]
            z_x = gen_noise(z_one_hot, flags, sym=False)  # [B, max_num_nodes, num_class_X]
            mean_x, std_x = self.sde_x.marGINal_prob(z_one_hot, t)  # [B, max_num_nodes, num_class_X], [B]
            perturbed_x = mean_x + std_x[:, None, None] * z_x  # [B, max_num_nodes, num_class_X]
        else:
            z = z.float().unsqueeze(2)
            z_x = gen_noise(z, flags, sym=False)  # [B, max_num_nodes, 1]
            mean_x, std_x = self.sde_x.marGINal_prob(z, t)  # [B, max_num_nodes, 1], [B]
            perturbed_x = mean_x + std_x[:, None, None] * z_x  # [B, max_num_nodes, 1]
        perturbed_x = mask_x(perturbed_x, flags)  # [B, max_num_nodes, num_class_X] or [B, max_num_nodes, 1]
        score_fn_x = self.get_score_fn(self.sde_x, self.node_score_network, train=train, continuous=continuous)  # function with output dim [B, max_num_nodes, num_class_X] or [B, max_num_nodes, 1]
        
        # 3Dx + pertubed atomx
        #perturbed_x = self.embedding_3D(node_3D_repr) + self.embedding_X(perturbed_x)  # [B, max_num_nodes, hdim]
        node_3D_repr = self.embedding_3D(node_3D_repr)
        perturbed_x = torch.cat([node_3D_repr,self.embedding_X(perturbed_x)],-1)  # [B, max_num_nodes, 2*hdim]
        score_adj = score_fn_adj(perturbed_x, perturbed_adj,node_3D_repr, flags, t)  # [B, max_num_nodes, max_num_nodes]
        score_x = score_fn_x(perturbed_x, perturbed_adj,node_3D_repr, flags, t)  # [B, max_num_nodes, num_class_X] or [B, max_num_nodes, 1]

        reduce_op = torch.mean if reduce_mean else lambda *args, **kwargs: 0.5 * torch.sum(*args, **kwargs)

        # TODO: we need to double-check this
        if anneal_power == 0:
            losses_x = torch.square(score_x + z_x)  # [B, max_num_nodes, num_class_X] or [B, max_num_nodes, 1]
            losses_adj = torch.square(score_adj + z_adj)  # [B, max_num_nodes, max_num_nodes]

        else:
            annealed_std_x = std_x ** anneal_power  # [B]
            annealed_std_x = annealed_std_x.unsqueeze(1).unsqueeze(2)  # [B, 1, 1]
            losses_x = torch.square(score_x + z_x) * annealed_std_x  # [B, max_num_nodes, num_class_X] or [B, max_num_nodes, 1]
            
            annealed_std_adj = std_adj ** anneal_power  # [B]
            annealed_std_adj = annealed_std_adj.unsqueeze(1).unsqueeze(2)  # [B, 1, 1]
            losses_adj = torch.square(score_adj + z_adj) * annealed_std_adj  # [B, max_num_nodes, max_num_nodes]

        losses_x = reduce_op(losses_x.reshape(losses_x.shape[0], -1), dim=-1)
        losses_adj = reduce_op(losses_adj.reshape(losses_adj.shape[0], -1), dim=-1)

        return torch.mean(losses_x), torch.mean(losses_adj)


def node_flags(adj, eps=1e-5):

    flags = torch.abs(adj).sum(-1).gt(eps).to(dtype=torch.float32)

    if len(flags.shape)==3:
        flags = flags[:,0,:]
    return flags


def gen_noise(x, flags, sym=True):
    z = torch.randn_like(x)
    if sym:
        z = z.triu(1)
        z = z + z.transpose(-1,-2)
        z = mask_adjs(z, flags)
    else:
        z = mask_x(z, flags)
    return z


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

# -------- Mask batch of node features with 0-1 flags tensor --------
def mask_x(x, flags):
    if flags is None:
        flags = torch.ones((x.shape[0], x.shape[1]), device=x.device)
    return x * flags[:,:,None]
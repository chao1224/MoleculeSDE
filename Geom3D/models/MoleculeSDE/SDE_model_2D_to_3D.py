import numpy as np
import torch
from torch import nn
from torch_scatter import scatter_mean
from .layers import MultiLayerPerceptron
from .equivariant_scorenetwork import EquivariantScoreNetwork
from .SDE_sparse import VPSDE, VESDE, subVPSDE


EPSILON = 1e-6


def get_beta_schedule(beta_schedule, *, beta_min, beta_max, num_diffusion_timesteps):
    def sigmoid(x):
        return 1 / (np.exp(-x) + 1)

    if beta_schedule == "quad":
        betas = (np.linspace(beta_min**0.5, beta_max**0.5, num_diffusion_timesteps, dtype=np.float64) ** 2)
    elif beta_schedule == "linear":
        betas = np.linspace(beta_min, beta_max, num_diffusion_timesteps, dtype=np.float64)
    elif beta_schedule == "const":
        betas = beta_max * np.ones(num_diffusion_timesteps, dtype=np.float64)
    elif beta_schedule == "jsd":
        betas = 1.0 / np.linspace(num_diffusion_timesteps, 1, num_diffusion_timesteps, dtype=np.float64)
    elif beta_schedule == "sigmoid":
        betas = np.linspace(-6, 6, num_diffusion_timesteps)
        betas = sigmoid(betas) * (beta_max - beta_min) + beta_min
    else:
        raise NotImplementedError(beta_schedule)
    assert betas.shape == (num_diffusion_timesteps,)
    betas = torch.from_numpy(betas).float()
    return betas


def coord2basis(pos, row, col):
    coord_diff = pos[row] - pos[col]
    radial = torch.sum((coord_diff) ** 2, 1).unsqueeze(1)
    coord_cross = torch.cross(pos[row], pos[col])

    norm = torch.sqrt(radial) + EPSILON
    coord_diff = coord_diff / norm
    cross_norm = torch.sqrt(torch.sum((coord_cross) ** 2, 1).unsqueeze(1)) + EPSILON
    coord_cross = coord_cross / cross_norm

    coord_vertical = torch.cross(coord_diff, coord_cross)

    return coord_diff, coord_cross, coord_vertical


def get_perturb_distance(p_pos, edge_index):
    pos = p_pos
    row, col = edge_index
    d = (pos[row] - pos[col]).norm(dim=-1).unsqueeze(-1)  # (num_edge, 1)
    return d


class GaussianFourierProjection(nn.Module):
    """Gaussian Fourier embeddings for noise levels."""

    def __init__(self, embedding_size, scale=1.0):
        super().__init__()
        self.W = nn.Parameter(torch.randn(embedding_size) * scale, requires_grad=False)

    def forward(self, x):
        x_proj = x * self.W[None, :] * 2 * np.pi
        return torch.cat([torch.sin(x_proj), torch.cos(x_proj)], dim=-1)


class SDEModel2Dto3D_01(torch.nn.Module):
    def __init__(
        self, emb_dim, hidden_dim,
        beta_schedule, beta_min, beta_max, num_diffusion_timesteps, SDE_type="VE",
        short_cut=False, concat_hidden=False, use_extend_graph=False):

        super(SDEModel2Dto3D_01, self).__init__()
        self.emb_dim = emb_dim
        self.hidden_dim = hidden_dim
        self.SDE_type = SDE_type
        self.use_extend_graph = use_extend_graph

        self.node_emb = MultiLayerPerceptron(self.emb_dim, [self.hidden_dim], activation="silu")
        self.edge_2D_emb = nn.Sequential(nn.Linear(self.emb_dim*2, self.emb_dim), nn.BatchNorm1d(self.emb_dim), nn.ReLU(), nn.Linear(self.emb_dim, self.hidden_dim))

        self.coff_gaussian_fourier = GaussianFourierProjection(embedding_size=self.hidden_dim, scale=1)
        self.coff_mlp = nn.Linear(4 * self.hidden_dim, self.hidden_dim)
        self.project = MultiLayerPerceptron(2 * self.hidden_dim + 2, [self.hidden_dim, self.hidden_dim], activation="silu")

        self.score_network = EquivariantScoreNetwork(hidden_dim=self.hidden_dim, hidden_coff_dim=128, activation="silu", short_cut=short_cut, concat_hidden=concat_hidden)

        if self.SDE_type in ["VE", "VE_test"]:
            self.sde_pos = VESDE(sigma_min=beta_min, sigma_max=beta_max, N=num_diffusion_timesteps)
        elif self.SDE_type in ["VP", "VP_test"]:
            self.sde_pos = VPSDE(beta_min=beta_min, beta_max=beta_max, N=num_diffusion_timesteps)
        elif self.SDE_type == "discrete_VE":
            betas = get_beta_schedule(
                beta_schedule=beta_schedule,
                beta_min=beta_min,
                beta_max=beta_max,
                num_diffusion_timesteps=num_diffusion_timesteps,
            )
            self.betas = nn.Parameter(betas, requires_grad=False)
            # variances
            alphas = (1. - betas).cumprod(dim=0)
            self.alphas = nn.Parameter(alphas, requires_grad=False)
            # print("betas used in 2D to 3D diffusion model", self.betas)
            # print("alphas used in 2D to 3D diffusion model", self.alphas)

        self.num_diffusion_timesteps = num_diffusion_timesteps
        return

    def get_embedding(self, coff_index):
        coff_embeds = []
        for i in [0, 2]:  # if i=1, then x=0
            coff_embeds.append(self.coff_gaussian_fourier(coff_index[:, i:i + 1]))  # [E, 2C]
        coff_embeds = torch.cat(coff_embeds, dim=-1)  # [E, 6C]
        coff_embeds = self.coff_mlp(coff_embeds)

        return coff_embeds

    def forward(self, node_2D_repr, data, anneal_power):
        pos = data.positions
        pos.requires_grad = True

        # data = self.get_distance(data)
        node2graph = data.batch
        if self.use_extend_graph:
            extended_edge_index = data.extended_edge_index
        else:
            extended_edge_index = data.edge_index

        # Perterb pos
        pos_noise = torch.randn_like(pos)

        # sample variances
        time_step = torch.randint(0, self.num_diffusion_timesteps, size=(data.num_graphs // 2 + 1,), device=pos.device)
        time_step = torch.cat([time_step, self.num_diffusion_timesteps - time_step - 1], dim=0)[:data.num_graphs]  # (num_graph, )

        if self.SDE_type in ["VE", "VP"]:
            time_step = time_step / self.num_diffusion_timesteps * (1 - EPSILON) + EPSILON  # normalize to [0, 1]
            time_step = time_step.squeeze(-1)
            t_pos = time_step.index_select(0, node2graph)  # (num_graph, )
            mean_pos, std_pos = self.sde_pos.marGINal_prob(pos, t_pos)
            pos_perturbed = mean_pos + std_pos[:, None] * pos_noise
            
        elif self.SDE_type in ["VE_test", "VP_test"]:
            time_step = time_step.squeeze(-1)
            t_pos = time_step.index_select(0, node2graph)  # (num_graph, )
            mean_pos, std_pos = self.sde_pos.marGINal_prob(pos, t_pos)
            pos_perturbed = mean_pos + std_pos[:, None] * pos_noise

        elif self.SDE_type == "discrete_VE":
            a = self.alphas.index_select(0, time_step)  # (num_graph, )
            a_pos = a.index_select(0, node2graph).unsqueeze(-1)  # (num_nodes, 1)
            pos_perturbed = pos + pos_noise * (1.0 - a_pos).sqrt() / a_pos.sqrt()
        
        # edge_attr from 2D represenattion node_2D_repr
        row, col = extended_edge_index
        edge_attr_2D = torch.cat([node_2D_repr[row], node_2D_repr[col]], dim=-1)
        edge_attr_2D = self.edge_2D_emb(edge_attr_2D)
        
        # construct geometric features
        coord_diff, coord_cross, coord_vertical = coord2basis(pos_perturbed, row, col)  # [num_edge, 3] * 3
        equivariant_basis = [coord_diff, coord_cross, coord_vertical]
        edge_basis = torch.cat([coord_diff.unsqueeze(1), coord_cross.unsqueeze(1), coord_vertical.unsqueeze(1)], dim=1)  # [num_edge, 3, 3]
        r_i, r_j = pos_perturbed[row], pos_perturbed[col]  # [num_edge, 3]
        coff_i = torch.matmul(edge_basis, r_i.unsqueeze(-1)).squeeze(-1)  # [num_edge, 3]
        coff_j = torch.matmul(edge_basis, r_j.unsqueeze(-1)).squeeze(-1)  # [num_edge, 3]
        coff_i[:, 1] = torch.abs(coff_i[:, 1].clone())
        coff_j[:, 1] = torch.abs(coff_j[:, 1].clone())
        coff_mul = coff_i * coff_j  # [num_edge, 3]
        coff_i_norm = coff_i.norm(dim=-1, keepdim=True) # [num_edge, 1]
        coff_j_norm = coff_j.norm(dim=-1, keepdim=True) # [num_edge, 1]
        pseudo_cos = coff_mul.sum(dim=-1, keepdim=True) / (coff_i_norm + EPSILON) / (coff_j_norm + EPSILON)
        pseudo_sin = torch.sqrt(1 - pseudo_cos ** 2)
        pseudo_angle = torch.cat([pseudo_sin, pseudo_cos], dim=-1)  # [num_edge, 2]
        embed_i = self.get_embedding(coff_i)  # [num_edge, C]
        embed_j = self.get_embedding(coff_j)  # [num_edge, C]
        edge_embed = torch.cat([pseudo_angle, embed_i, embed_j], dim=-1)
        edge_attr_3D_frame_invariant = self.project(edge_embed)
        
        edge_attr = edge_attr_2D + edge_attr_3D_frame_invariant

        # match dimension
        node_attr = self.node_emb(node_2D_repr)

        # estimate scores
        output = self.score_network(extended_edge_index, node_attr, edge_attr, equivariant_basis)
        scores = output["gradient"]
        if anneal_power == 0:
            loss_pos = torch.sum((scores - pos_noise) ** 2, -1)  # (num_node)
        else:
            annealed_std = std_pos ** anneal_power  # (num_node)
            annealed_std = annealed_std.unsqueeze(1,)  # (num_node,1)
            loss_pos = torch.sum((scores - pos_noise) ** 2 * annealed_std, -1)  # (num_node)
        loss_pos = scatter_mean(loss_pos, node2graph)  # (num_graph)

        loss_dict = {
            'position': loss_pos.mean(),
        }
        return loss_dict

    @torch.no_grad()
    def get_score(self, node_2D_repr, data, pos_perturbed, sigma, t_pos):
        node_attr = self.node_emb(node_2D_repr)
        
        if self.use_extend_graph:
            extended_edge_index = data.extended_edge_index
        else:
            extended_edge_index = data.edge_index
        
        # edge_attr from 2D represenattion node_2D_repr
        row, col = extended_edge_index        
        edge_attr_2D = torch.cat([node_2D_repr[row], node_2D_repr[col]], dim=-1)
        edge_attr_2D = self.edge_2D_emb(edge_attr_2D)
        
        # construct geometric features
        coord_diff, coord_cross, coord_vertical = coord2basis(pos_perturbed, row, col)  # [num_edge, 3] * 3
        equivariant_basis = [coord_diff, coord_cross, coord_vertical]
        edge_basis = torch.cat([coord_diff.unsqueeze(1), coord_cross.unsqueeze(1), coord_vertical.unsqueeze(1)], dim=1)  # [num_edge, 3, 3]
        r_i, r_j = pos_perturbed[row], pos_perturbed[col]  # [num_edge, 3]
        coff_i = torch.matmul(edge_basis, r_i.unsqueeze(-1)).squeeze(-1)  # [num_edge, 3]
        coff_j = torch.matmul(edge_basis, r_j.unsqueeze(-1)).squeeze(-1)  # [num_edge, 3]
        coff_i[:, 1] = torch.abs(coff_i[:, 1].clone())
        coff_j[:, 1] = torch.abs(coff_j[:, 1].clone())
        coff_mul = coff_i * coff_j  # [num_edge, 3]
        coff_i_norm = coff_i.norm(dim=-1, keepdim=True) # [num_edge, 1]
        coff_j_norm = coff_j.norm(dim=-1, keepdim=True) # [num_edge, 1]
        pseudo_cos = coff_mul.sum(dim=-1, keepdim=True) / (coff_i_norm + EPSILON) / (coff_j_norm + EPSILON)
        pseudo_sin = torch.sqrt(1 - pseudo_cos ** 2)
        pseudo_angle = torch.cat([pseudo_sin, pseudo_cos], dim=-1)  # [num_edge, 2]
        embed_i = self.get_embedding(coff_i)  # [num_edge, C]
        embed_j = self.get_embedding(coff_j)  # [num_edge, C]
        edge_embed = torch.cat([pseudo_angle, embed_i, embed_j], dim=-1)
        edge_attr_3D_frame_invariant = self.project(edge_embed)
        
        edge_attr = edge_attr_2D + edge_attr_3D_frame_invariant
        
        # match dimension
        node_attr = self.node_emb(node_2D_repr)
        
        # estimate scores
        output = self.score_network(extended_edge_index, node_attr, edge_attr, equivariant_basis)
        output = output["gradient"]
        scores = -output

        _, std_pos = self.sde_pos.marGINal_prob(pos_perturbed, t_pos)
        scores = scores / std_pos[:, None]
        # print(t_pos, std_pos)
        return scores


class SDEModel2Dto3D_02(torch.nn.Module):
    def __init__(
        self, emb_dim, hidden_dim,
        beta_schedule, beta_min, beta_max, num_diffusion_timesteps, SDE_type="VE",
        short_cut=False, concat_hidden=False, use_extend_graph=False):

        super(SDEModel2Dto3D_02, self).__init__()
        self.emb_dim = emb_dim
        self.hidden_dim = hidden_dim
        self.SDE_type = SDE_type
        self.use_extend_graph = use_extend_graph

        self.node_emb = MultiLayerPerceptron(self.emb_dim, [self.hidden_dim], activation="silu")
        self.edge_2D_emb = nn.Sequential(nn.Linear(self.emb_dim*2, self.emb_dim), nn.BatchNorm1d(self.emb_dim), nn.ReLU(), nn.Linear(self.emb_dim, self.hidden_dim))

        self.dist_gaussian_fourier = GaussianFourierProjection(embedding_size=self.hidden_dim, scale=1)
        self.input_mlp = MultiLayerPerceptron(2*self.hidden_dim, [self.hidden_dim], activation="silu")

        self.coff_gaussian_fourier = GaussianFourierProjection(embedding_size=self.hidden_dim, scale=1)
        self.coff_mlp = nn.Linear(4 * self.hidden_dim, self.hidden_dim)
        self.project = MultiLayerPerceptron(2 * self.hidden_dim + 2, [self.hidden_dim, self.hidden_dim], activation="silu")

        self.score_network = EquivariantScoreNetwork(hidden_dim=self.hidden_dim, hidden_coff_dim=128, activation="silu", short_cut=short_cut, concat_hidden=concat_hidden)

        if self.SDE_type in ["VE", "VE_test"]:
            self.sde_pos = VESDE(sigma_min=beta_min, sigma_max=beta_max, N=num_diffusion_timesteps)
        elif self.SDE_type in ["VP", "VP_test"]:
            self.sde_pos = VPSDE(beta_min=beta_min, beta_max=beta_max, N=num_diffusion_timesteps)
        elif self.SDE_type == "discrete_VE":
            betas = get_beta_schedule(
                beta_schedule=beta_schedule,
                beta_min=beta_min,
                beta_max=beta_max,
                num_diffusion_timesteps=num_diffusion_timesteps,
            )
            self.betas = nn.Parameter(betas, requires_grad=False)
            # variances
            alphas = (1. - betas).cumprod(dim=0)
            self.alphas = nn.Parameter(alphas, requires_grad=False)
            # print("betas used in 2D to 3D diffusion model", self.betas)
            # print("alphas used in 2D to 3D diffusion model", self.alphas)

        self.num_diffusion_timesteps = num_diffusion_timesteps
        return

    def get_embedding(self, coff_index):
        coff_embeds = []
        for i in [0, 2]:  # if i=1, then x=0
            coff_embeds.append(self.coff_gaussian_fourier(coff_index[:, i:i + 1]))  # [E, 2C]
        coff_embeds = torch.cat(coff_embeds, dim=-1)  # [E, 6C]
        coff_embeds = self.coff_mlp(coff_embeds)

        return coff_embeds

    def forward(self, node_2D_repr, data, anneal_power):
        pos = data.positions
        pos.requires_grad = True

        # data = self.get_distance(data)
        node2graph = data.batch
        if self.use_extend_graph:
            extended_edge_index = data.extended_edge_index
        else:
            extended_edge_index = data.edge_index
        
        # Perterb pos
        pos_noise = torch.randn_like(pos)

        # sample variances
        time_step = torch.randint(0, self.num_diffusion_timesteps, size=(data.num_graphs // 2 + 1,), device=pos.device)
        time_step = torch.cat([time_step, self.num_diffusion_timesteps - time_step - 1], dim=0)[:data.num_graphs]  # (num_graph, )

        if self.SDE_type in ["VE", "VP"]:
            time_step = time_step / self.num_diffusion_timesteps * (1 - EPSILON) + EPSILON  # normalize to [0, 1]
            time_step = time_step.squeeze(-1)
            t_pos = time_step.index_select(0, node2graph)  # (num_nodes, )
            mean_pos, std_pos = self.sde_pos.marGINal_prob(pos, t_pos)
            pos_perturbed = mean_pos + std_pos[:, None] * pos_noise
            
        elif self.SDE_type in ["VE_test", "VP_test"]:
            time_step = time_step.squeeze(-1)
            t_pos = time_step.index_select(0, node2graph)  # (num_graph, )
            mean_pos, std_pos = self.sde_pos.marGINal_prob(pos, t_pos)
            pos_perturbed = mean_pos + std_pos[:, None] * pos_noise

        elif self.SDE_type == "discrete_VE":
            a = self.alphas.index_select(0, time_step)  # (num_graph, )
            a_pos = a.index_select(0, node2graph).unsqueeze(-1)  # (num_nodes, 1)
            pos_perturbed = pos + pos_noise * (1.0 - a_pos).sqrt() / a_pos.sqrt()
        
        distance_perturbed = get_perturb_distance(pos_perturbed, extended_edge_index)

        # edge_attr should come from 2D represenattion x
        row, col = extended_edge_index
        edge_attr_2D = torch.cat([node_2D_repr[row], node_2D_repr[col]], dim=-1)
        edge_attr_2D = self.edge_2D_emb(edge_attr_2D)
        
        distance_perturbed_emb = self.dist_gaussian_fourier(distance_perturbed)  # (num_edge, hidden*2)
        edge_attr_3D_invariant = self.input_mlp(distance_perturbed_emb)  # (num_edge, hidden)

        # construct geometric features
        coord_diff, coord_cross, coord_vertical = coord2basis(pos_perturbed, row, col)  # [num_edge, 3] * 3
        equivariant_basis = [coord_diff, coord_cross, coord_vertical]
        edge_basis = torch.cat([coord_diff.unsqueeze(1), coord_cross.unsqueeze(1), coord_vertical.unsqueeze(1)], dim=1)  # [num_edge, 3, 3]
        r_i, r_j = pos_perturbed[row], pos_perturbed[col]  # [num_edge, 3]
        coff_i = torch.matmul(edge_basis, r_i.unsqueeze(-1)).squeeze(-1)  # [num_edge, 3]
        coff_j = torch.matmul(edge_basis, r_j.unsqueeze(-1)).squeeze(-1)  # [num_edge, 3]
        coff_i[:, 1] = torch.abs(coff_i[:, 1].clone())
        coff_j[:, 1] = torch.abs(coff_j[:, 1].clone())
        coff_mul = coff_i * coff_j  # [num_edge, 3]
        coff_i_norm = coff_i.norm(dim=-1, keepdim=True) # [num_edge, 1]
        coff_j_norm = coff_j.norm(dim=-1, keepdim=True) # [num_edge, 1]
        pseudo_cos = coff_mul.sum(dim=-1, keepdim=True) / (coff_i_norm + EPSILON) / (coff_j_norm + EPSILON)
        pseudo_sin = torch.sqrt(1 - pseudo_cos ** 2)
        pseudo_angle = torch.cat([pseudo_sin, pseudo_cos], dim=-1)
        embed_i = self.get_embedding(coff_i)  # [num_edge, C]
        embed_j = self.get_embedding(coff_j)  # [num_edge, C]
        edge_embed = torch.cat([pseudo_angle, embed_i, embed_j], dim=-1)
        edge_attr_3D_frame_invariant = self.project(edge_embed)
        
        edge_attr = edge_attr_3D_invariant * edge_attr_2D + edge_attr_3D_frame_invariant

        # match dimension
        node_attr = self.node_emb(node_2D_repr)

        # estimate scores
        output = self.score_network(extended_edge_index, node_attr, edge_attr, equivariant_basis)
        scores = output["gradient"]
        if anneal_power == 0:
            loss_pos = torch.sum((scores - pos_noise) ** 2, -1)  # (num_node)
        else:
            annealed_std = std_pos ** anneal_power  # (num_node)
            annealed_std = annealed_std.unsqueeze(1,)  # (num_node,1)
            loss_pos = torch.sum((scores - pos_noise) ** 2 * annealed_std, -1)  # (num_node)
        loss_pos = scatter_mean(loss_pos, node2graph)  # (num_graph)

        loss_dict = {
            'position': loss_pos.mean(),
        }
        return loss_dict

    @torch.no_grad()
    def get_score(self, node_2D_repr, data, pos_perturbed, sigma, t_pos):
        node_attr = self.node_emb(node_2D_repr)
        
        if self.use_extend_graph:
            extended_edge_index = data.extended_edge_index
        else:
            extended_edge_index = data.edge_index
        
        distance_perturbed = get_perturb_distance(pos_perturbed, extended_edge_index)

        # edge_attr from 2D represenattion node_2D_repr
        row, col = extended_edge_index        
        edge_attr_2D = torch.cat([node_2D_repr[row], node_2D_repr[col]], dim=-1)
        edge_attr_2D = self.edge_2D_emb(edge_attr_2D)
        
        distance_perturbed_emb = self.dist_gaussian_fourier(distance_perturbed)  # (num_edge, hidden*2)
        edge_attr_3D_invariant = self.input_mlp(distance_perturbed_emb)  # (num_edge, hidden)

        # construct geometric features
        coord_diff, coord_cross, coord_vertical = coord2basis(pos_perturbed, row, col)  # [num_edge, 3] * 3
        equivariant_basis = [coord_diff, coord_cross, coord_vertical]
        edge_basis = torch.cat([coord_diff.unsqueeze(1), coord_cross.unsqueeze(1), coord_vertical.unsqueeze(1)], dim=1)  # [num_edge, 3, 3]
        r_i, r_j = pos_perturbed[row], pos_perturbed[col]  # [num_edge, 3]
        coff_i = torch.matmul(edge_basis, r_i.unsqueeze(-1)).squeeze(-1)  # [num_edge, 3]
        coff_j = torch.matmul(edge_basis, r_j.unsqueeze(-1)).squeeze(-1)  # [num_edge, 3]
        coff_i[:, 1] = torch.abs(coff_i[:, 1].clone())
        coff_j[:, 1] = torch.abs(coff_j[:, 1].clone())
        coff_mul = coff_i * coff_j  # [num_edge, 3]
        coff_i_norm = coff_i.norm(dim=-1, keepdim=True) # [num_edge, 1]
        coff_j_norm = coff_j.norm(dim=-1, keepdim=True) # [num_edge, 1]
        pseudo_cos = coff_mul.sum(dim=-1, keepdim=True) / (coff_i_norm + EPSILON) / (coff_j_norm + EPSILON)
        pseudo_sin = torch.sqrt(1 - pseudo_cos ** 2)
        pseudo_angle = torch.cat([pseudo_sin, pseudo_cos], dim=-1)
        embed_i = self.get_embedding(coff_i)  # [num_edge, C]
        embed_j = self.get_embedding(coff_j)  # [num_edge, C]
        edge_embed = torch.cat([pseudo_angle, embed_i, embed_j], dim=-1)
        edge_attr_3D_frame_invariant = self.project(edge_embed)
        
        edge_attr = edge_attr_3D_invariant * edge_attr_2D + edge_attr_3D_frame_invariant
        
        # match dimension
        node_attr = self.node_emb(node_2D_repr)
        
        # estimate scores
        output = self.score_network(extended_edge_index, node_attr, edge_attr, equivariant_basis)
        output = output["gradient"]
        scores = -output

        _, std_pos = self.sde_pos.marGINal_prob(pos_perturbed, t_pos)
        scores = scores / std_pos[:, None]
        # print(t_pos, std_pos)
        return scores


class SDEModel2Dto3D_03(torch.nn.Module):
    def __init__(
        self, emb_dim, hidden_dim,
        beta_schedule, beta_min, beta_max, num_diffusion_timesteps, SDE_type="VE",
        short_cut=False, concat_hidden=False, use_extend_graph=False):

        super(SDEModel2Dto3D_03, self).__init__()
        self.emb_dim = emb_dim
        self.hidden_dim = hidden_dim
        self.SDE_type = SDE_type
        self.use_extend_graph = use_extend_graph

        self.node_emb = MultiLayerPerceptron(self.emb_dim, [self.hidden_dim], activation="silu")
        self.edge_2D_emb = nn.Sequential(nn.Linear(self.emb_dim*2, self.emb_dim), nn.BatchNorm1d(self.emb_dim), nn.ReLU(), nn.Linear(self.emb_dim, self.hidden_dim))
        self.edge_2D_emb = nn.Linear(self.emb_dim*2, self.hidden_dim)
        # TODO: will hack
        self.edge_emb = torch.nn.Embedding(100, self.hidden_dim)

        self.coff_gaussian_fourier = GaussianFourierProjection(embedding_size=self.hidden_dim, scale=1)
        self.coff_mlp = nn.Linear(4 * self.hidden_dim, self.hidden_dim)
        self.project = MultiLayerPerceptron(2 * self.hidden_dim + 2, [self.hidden_dim, self.hidden_dim], activation="silu")

        self.score_network = EquivariantScoreNetwork(hidden_dim=self.hidden_dim, hidden_coff_dim=128, activation="silu", short_cut=short_cut, concat_hidden=concat_hidden)

        if self.SDE_type in ["VE", "VE_test"]:
            self.sde_pos = VESDE(sigma_min=beta_min, sigma_max=beta_max, N=num_diffusion_timesteps)
        elif self.SDE_type in ["VP", "VP_test"]:
            self.sde_pos = VPSDE(beta_min=beta_min, beta_max=beta_max, N=num_diffusion_timesteps)
        elif self.SDE_type == "discrete_VE":
            betas = get_beta_schedule(
                beta_schedule=beta_schedule,
                beta_min=beta_min,
                beta_max=beta_max,
                num_diffusion_timesteps=num_diffusion_timesteps,
            )
            self.betas = nn.Parameter(betas, requires_grad=False)
            # variances
            alphas = (1. - betas).cumprod(dim=0)
            self.alphas = nn.Parameter(alphas, requires_grad=False)
            # print("betas used in 2D to 3D diffusion model", self.betas)
            # print("alphas used in 2D to 3D diffusion model", self.alphas)

        self.num_diffusion_timesteps = num_diffusion_timesteps
        return

    def get_embedding(self, coff_index):
        coff_embeds = []
        for i in [0, 2]:  # if i=1, then x=0
            coff_embeds.append(self.coff_gaussian_fourier(coff_index[:, i:i + 1]))  # [E, 2C]
        coff_embeds = torch.cat(coff_embeds, dim=-1)  # [E, 6C]
        coff_embeds = self.coff_mlp(coff_embeds)

        return coff_embeds

    def forward(self, node_2D_repr, data, anneal_power):
        pos = data.positions
        pos.requires_grad = True

        node2graph = data.batch
        if self.use_extend_graph:
            extended_edge_index = data.extended_edge_index
        else:
            extended_edge_index = data.edge_index   

        # Perterb pos
        pos_noise = torch.randn_like(pos)

        # sample variances
        time_step = torch.randint(0, self.num_diffusion_timesteps, size=(data.num_graphs // 2 + 1,), device=pos.device)
        time_step = torch.cat([time_step, self.num_diffusion_timesteps - time_step - 1], dim=0)[:data.num_graphs]  # (num_graph, )

        if self.SDE_type in ["VE", "VP"]:
            time_step = time_step / self.num_diffusion_timesteps * (1 - EPSILON) + EPSILON  # normalize to [0, 1]
            time_step = time_step.squeeze(-1)
            t_pos = time_step.index_select(0, node2graph)  # (num_graph, )
            mean_pos, std_pos = self.sde_pos.marGINal_prob(pos, t_pos)
            pos_perturbed = mean_pos + std_pos[:, None] * pos_noise
            
        elif self.SDE_type in ["VE_test", "VP_test"]:
            time_step = time_step.squeeze(-1)
            t_pos = time_step.index_select(0, node2graph)  # (num_graph, )
            mean_pos, std_pos = self.sde_pos.marGINal_prob(pos, t_pos)
            pos_perturbed = mean_pos + std_pos[:, None] * pos_noise

        elif self.SDE_type == "discrete_VE":
            a = self.alphas.index_select(0, time_step)  # (num_graph, )
            a_pos = a.index_select(0, node2graph).unsqueeze(-1)  # (num_nodes, 1)
            pos_perturbed = pos + pos_noise * (1.0 - a_pos).sqrt() / a_pos.sqrt()
        
        # edge_attr from 2D represenattion node_2D_repr
        row, col = extended_edge_index
        edge_attr_2D = torch.cat([node_2D_repr[row], node_2D_repr[col]], dim=-1)
        edge_attr_input = self.edge_emb(data.extended_edge_attr) # (num_edge, hidden)
        edge_attr_2D = self.edge_2D_emb(edge_attr_2D) + edge_attr_input
        
        # construct geometric features
        coord_diff, coord_cross, coord_vertical = coord2basis(pos_perturbed, row, col)  # [num_edge, 3] * 3
        equivariant_basis = [coord_diff, coord_cross, coord_vertical]
        edge_basis = torch.cat([coord_diff.unsqueeze(1), coord_cross.unsqueeze(1), coord_vertical.unsqueeze(1)], dim=1)  # [num_edge, 3, 3]
        r_i, r_j = pos_perturbed[row], pos_perturbed[col]  # [num_edge, 3]
        coff_i = torch.matmul(edge_basis, r_i.unsqueeze(-1)).squeeze(-1)  # [num_edge, 3]
        coff_j = torch.matmul(edge_basis, r_j.unsqueeze(-1)).squeeze(-1)  # [num_edge, 3]
        coff_i[:, 1] = torch.abs(coff_i[:, 1].clone())
        coff_j[:, 1] = torch.abs(coff_j[:, 1].clone())
        coff_mul = coff_i * coff_j  # [num_edge, 3]
        coff_i_norm = coff_i.norm(dim=-1, keepdim=True) # [num_edge, 1]
        coff_j_norm = coff_j.norm(dim=-1, keepdim=True) # [num_edge, 1]
        pseudo_cos = coff_mul.sum(dim=-1, keepdim=True) / (coff_i_norm + EPSILON) / (coff_j_norm + EPSILON)
        pseudo_sin = torch.sqrt(1 - pseudo_cos ** 2)
        pseudo_angle = torch.cat([pseudo_sin, pseudo_cos], dim=-1)  # [num_edge, 2]
        embed_i = self.get_embedding(coff_i)  # [num_edge, C]
        embed_j = self.get_embedding(coff_j)  # [num_edge, C]
        edge_embed = torch.cat([pseudo_angle, embed_i, embed_j], dim=-1)
        edge_attr_3D_frame_invariant = self.project(edge_embed)
        
        edge_attr = edge_attr_2D + edge_attr_3D_frame_invariant

        # match dimension
        node_attr = self.node_emb(node_2D_repr)

        # estimate scores
        output = self.score_network(extended_edge_index, node_attr, edge_attr, equivariant_basis)
        scores = output["gradient"]
        if anneal_power == 0:
            loss_pos = torch.sum((scores - pos_noise) ** 2, -1)  # (num_node)
        else:
            annealed_std = std_pos ** anneal_power  # (num_node)
            annealed_std = annealed_std.unsqueeze(1,)  # (num_node,1)
            loss_pos = torch.sum((scores - pos_noise) ** 2 * annealed_std, -1)  # (num_node)
        loss_pos = scatter_mean(loss_pos, node2graph)  # (num_graph)

        loss_dict = {
            'position': loss_pos.mean(),
        }
        return loss_dict

    @torch.no_grad()
    def get_score(self, node_2D_repr, data, pos_perturbed, sigma, t_pos):
        node_attr = self.node_emb(node_2D_repr)
        
        if self.use_extend_graph:
            extended_edge_index = data.extended_edge_index
        else:
            extended_edge_index = data.edge_index
        
        # edge_attr from 2D represenattion node_2D_repr
        row, col = extended_edge_index        
        edge_attr_2D = torch.cat([node_2D_repr[row], node_2D_repr[col]], dim=-1)
        edge_attr_input = self.edge_emb(data.extended_edge_attr) # (num_edge, hidden)   
        edge_attr_2D = self.edge_2D_emb(edge_attr_2D) + edge_attr_input
        
        # construct geometric features
        coord_diff, coord_cross, coord_vertical = coord2basis(pos_perturbed, row, col)  # [num_edge, 3] * 3
        equivariant_basis = [coord_diff, coord_cross, coord_vertical]
        edge_basis = torch.cat([coord_diff.unsqueeze(1), coord_cross.unsqueeze(1), coord_vertical.unsqueeze(1)], dim=1)  # [num_edge, 3, 3]
        r_i, r_j = pos_perturbed[row], pos_perturbed[col]  # [num_edge, 3]
        coff_i = torch.matmul(edge_basis, r_i.unsqueeze(-1)).squeeze(-1)  # [num_edge, 3]
        coff_j = torch.matmul(edge_basis, r_j.unsqueeze(-1)).squeeze(-1)  # [num_edge, 3]
        coff_i[:, 1] = torch.abs(coff_i[:, 1].clone())
        coff_j[:, 1] = torch.abs(coff_j[:, 1].clone())
        coff_mul = coff_i * coff_j  # [num_edge, 3]
        coff_i_norm = coff_i.norm(dim=-1, keepdim=True) # [num_edge, 1]
        coff_j_norm = coff_j.norm(dim=-1, keepdim=True) # [num_edge, 1]
        pseudo_cos = coff_mul.sum(dim=-1, keepdim=True) / (coff_i_norm + EPSILON) / (coff_j_norm + EPSILON)
        pseudo_sin = torch.sqrt(1 - pseudo_cos ** 2)
        pseudo_angle = torch.cat([pseudo_sin, pseudo_cos], dim=-1)  # [num_edge, 2]
        embed_i = self.get_embedding(coff_i)  # [num_edge, C]
        embed_j = self.get_embedding(coff_j)  # [num_edge, C]
        edge_embed = torch.cat([pseudo_angle, embed_i, embed_j], dim=-1)
        edge_attr_3D_frame_invariant = self.project(edge_embed)
        
        edge_attr = edge_attr_2D + edge_attr_3D_frame_invariant
        
        # match dimension
        node_attr = self.node_emb(node_2D_repr)
        
        # estimate scores
        output = self.score_network(extended_edge_index, node_attr, edge_attr, equivariant_basis)
        output = output["gradient"]
        scores = -output

        _, std_pos = self.sde_pos.marGINal_prob(pos_perturbed, t_pos)
        scores = scores / std_pos[:, None]
        # print(t_pos, std_pos)
        return scores


class SDEModel2Dto3D_04(torch.nn.Module):
    def __init__(
        self, emb_dim, hidden_dim,
        beta_schedule, beta_min, beta_max, num_diffusion_timesteps, SDE_type="VE",
        short_cut=False, concat_hidden=False, use_extend_graph=False):

        super(SDEModel2Dto3D_04, self).__init__()
        self.emb_dim = emb_dim
        self.hidden_dim = hidden_dim
        self.SDE_type = SDE_type
        self.use_extend_graph = use_extend_graph

        self.node_emb = MultiLayerPerceptron(self.emb_dim, [self.hidden_dim], activation="silu")
        self.edge_2D_emb = nn.Sequential(nn.Linear(self.emb_dim*2, self.emb_dim), nn.BatchNorm1d(self.emb_dim), nn.ReLU(), nn.Linear(self.emb_dim, self.hidden_dim))
        self.edge_2D_emb = nn.Linear(self.emb_dim*2, self.hidden_dim)
        # TODO: will hack
        self.edge_emb = torch.nn.Embedding(100, self.hidden_dim)

        self.coff_gaussian_fourier = GaussianFourierProjection(embedding_size=self.hidden_dim, scale=1)
        self.coff_mlp = nn.Linear(4 * self.hidden_dim, self.hidden_dim)
        self.project = MultiLayerPerceptron(2 * self.hidden_dim + 2, [self.hidden_dim, self.hidden_dim], activation="silu")

        self.score_network = EquivariantScoreNetwork(hidden_dim=self.hidden_dim, hidden_coff_dim=128, activation="silu", short_cut=short_cut, concat_hidden=concat_hidden)

        if self.SDE_type in ["VE", "VE_test"]:
            self.sde_pos = VESDE(sigma_min=beta_min, sigma_max=beta_max, N=num_diffusion_timesteps)
        elif self.SDE_type in ["VP", "VP_test"]:
            self.sde_pos = VPSDE(beta_min=beta_min, beta_max=beta_max, N=num_diffusion_timesteps)
        elif self.SDE_type == "discrete_VE":
            betas = get_beta_schedule(
                beta_schedule=beta_schedule,
                beta_min=beta_min,
                beta_max=beta_max,
                num_diffusion_timesteps=num_diffusion_timesteps,
            )
            self.betas = nn.Parameter(betas, requires_grad=False)
            # variances
            alphas = (1. - betas).cumprod(dim=0)
            self.alphas = nn.Parameter(alphas, requires_grad=False)
            # print("betas used in 2D to 3D diffusion model", self.betas)
            # print("alphas used in 2D to 3D diffusion model", self.alphas)

        self.num_diffusion_timesteps = num_diffusion_timesteps
        return

    def get_embedding(self, coff_index):
        coff_embeds = []
        for i in [0, 2]:  # if i=1, then x=0
            coff_embeds.append(self.coff_gaussian_fourier(coff_index[:, i:i + 1]))  # [E, 2C]
        coff_embeds = torch.cat(coff_embeds, dim=-1)  # [E, 6C]
        coff_embeds = self.coff_mlp(coff_embeds)

        return coff_embeds

    def forward(self, node_2D_repr, data, anneal_power):
        pos = data.positions
        pos.requires_grad = True

        node2graph = data.batch
        if self.use_extend_graph:
            extended_edge_index = data.extended_edge_index
        else:
            extended_edge_index = data.edge_index   

        # Perterb pos
        pos_noise = torch.randn_like(pos)

        # sample variances
        time_step = torch.randint(0, self.num_diffusion_timesteps, size=(data.num_graphs // 2 + 1,), device=pos.device)
        time_step = torch.cat([time_step, self.num_diffusion_timesteps - time_step - 1], dim=0)[:data.num_graphs]  # (num_graph, )

        if self.SDE_type in ["VE", "VP"]:
            time_step = time_step / self.num_diffusion_timesteps * (1 - EPSILON) + EPSILON  # normalize to [0, 1]
            time_step = time_step.squeeze(-1)
            t_pos = time_step.index_select(0, node2graph)  # (num_graph, )
            mean_pos, std_pos = self.sde_pos.marGINal_prob(pos, t_pos)
            pos_perturbed = mean_pos + std_pos[:, None] * pos_noise
            
        elif self.SDE_type in ["VE_test", "VP_test"]:
            time_step = time_step.squeeze(-1)
            t_pos = time_step.index_select(0, node2graph)  # (num_graph, )
            mean_pos, std_pos = self.sde_pos.marGINal_prob(pos, t_pos)
            pos_perturbed = mean_pos + std_pos[:, None] * pos_noise

        elif self.SDE_type == "discrete_VE":
            a = self.alphas.index_select(0, time_step)  # (num_graph, )
            a_pos = a.index_select(0, node2graph).unsqueeze(-1)  # (num_nodes, 1)
            pos_perturbed = pos + pos_noise * (1.0 - a_pos).sqrt() / a_pos.sqrt()
        
        # edge_attr from 2D represenattion node_2D_repr
        row, col = extended_edge_index
        edge_attr_2D = torch.cat([node_2D_repr[row] * node_2D_repr[col], node_2D_repr[row] + node_2D_repr[col]], dim=-1)
        edge_attr_input = self.edge_emb(data.extended_edge_attr) # (num_edge, hidden)
        edge_attr_2D = self.edge_2D_emb(edge_attr_2D) + edge_attr_input
        
        # construct geometric features
        coord_diff, coord_cross, coord_vertical = coord2basis(pos_perturbed, row, col)  # [num_edge, 3] * 3
        equivariant_basis = [coord_diff, coord_cross, coord_vertical]
        edge_basis = torch.cat([coord_diff.unsqueeze(1), coord_cross.unsqueeze(1), coord_vertical.unsqueeze(1)], dim=1)  # [num_edge, 3, 3]
        r_i, r_j = pos_perturbed[row], pos_perturbed[col]  # [num_edge, 3]
        coff_i = torch.matmul(edge_basis, r_i.unsqueeze(-1)).squeeze(-1)  # [num_edge, 3]
        coff_j = torch.matmul(edge_basis, r_j.unsqueeze(-1)).squeeze(-1)  # [num_edge, 3]
        coff_i[:, 1] = torch.abs(coff_i[:, 1].clone())
        coff_j[:, 1] = torch.abs(coff_j[:, 1].clone())
        coff_mul = coff_i * coff_j  # [num_edge, 3]
        coff_i_norm = coff_i.norm(dim=-1, keepdim=True) # [num_edge, 1]
        coff_j_norm = coff_j.norm(dim=-1, keepdim=True) # [num_edge, 1]
        pseudo_cos = coff_mul.sum(dim=-1, keepdim=True) / (coff_i_norm + EPSILON) / (coff_j_norm + EPSILON)
        pseudo_sin = torch.sqrt(1 - pseudo_cos ** 2)
        pseudo_angle = torch.cat([pseudo_sin, pseudo_cos], dim=-1)  # [num_edge, 2]
        embed_i = self.get_embedding(coff_i)  # [num_edge, C]
        embed_j = self.get_embedding(coff_j)  # [num_edge, C]
        edge_embed = torch.cat([pseudo_angle, embed_i, embed_j], dim=-1)
        edge_attr_3D_frame_invariant = self.project(edge_embed)
        
        edge_attr = edge_attr_2D + edge_attr_3D_frame_invariant

        # match dimension
        node_attr = self.node_emb(node_2D_repr)

        # estimate scores
        output = self.score_network(extended_edge_index, node_attr, edge_attr, equivariant_basis)
        scores = output["gradient"]
        if anneal_power == 0:
            loss_pos = torch.sum((scores - pos_noise) ** 2, -1)  # (num_node)
        else:
            annealed_std = std_pos ** anneal_power  # (num_node)
            annealed_std = annealed_std.unsqueeze(1,)  # (num_node,1)
            loss_pos = torch.sum((scores - pos_noise) ** 2 * annealed_std, -1)  # (num_node)
        loss_pos = scatter_mean(loss_pos, node2graph)  # (num_graph)

        loss_dict = {
            'position': loss_pos.mean(),
        }
        return loss_dict

    @torch.no_grad()
    def get_score(self, node_2D_repr, data, pos_perturbed, sigma, t_pos):
        node_attr = self.node_emb(node_2D_repr)
        
        if self.use_extend_graph:
            extended_edge_index = data.extended_edge_index
        else:
            extended_edge_index = data.edge_index
        
        # edge_attr from 2D represenattion node_2D_repr
        row, col = extended_edge_index        
        edge_attr_2D = torch.cat([node_2D_repr[row] * node_2D_repr[col], node_2D_repr[row] + node_2D_repr[col]], dim=-1)
        edge_attr_input = self.edge_emb(data.extended_edge_attr) # (num_edge, hidden)   
        edge_attr_2D = self.edge_2D_emb(edge_attr_2D) + edge_attr_input
        
        # construct geometric features
        coord_diff, coord_cross, coord_vertical = coord2basis(pos_perturbed, row, col)  # [num_edge, 3] * 3
        equivariant_basis = [coord_diff, coord_cross, coord_vertical]
        edge_basis = torch.cat([coord_diff.unsqueeze(1), coord_cross.unsqueeze(1), coord_vertical.unsqueeze(1)], dim=1)  # [num_edge, 3, 3]
        r_i, r_j = pos_perturbed[row], pos_perturbed[col]  # [num_edge, 3]
        coff_i = torch.matmul(edge_basis, r_i.unsqueeze(-1)).squeeze(-1)  # [num_edge, 3]
        coff_j = torch.matmul(edge_basis, r_j.unsqueeze(-1)).squeeze(-1)  # [num_edge, 3]
        coff_i[:, 1] = torch.abs(coff_i[:, 1].clone())
        coff_j[:, 1] = torch.abs(coff_j[:, 1].clone())
        coff_mul = coff_i * coff_j  # [num_edge, 3]
        coff_i_norm = coff_i.norm(dim=-1, keepdim=True) # [num_edge, 1]
        coff_j_norm = coff_j.norm(dim=-1, keepdim=True) # [num_edge, 1]
        pseudo_cos = coff_mul.sum(dim=-1, keepdim=True) / (coff_i_norm + EPSILON) / (coff_j_norm + EPSILON)
        pseudo_sin = torch.sqrt(1 - pseudo_cos ** 2)
        pseudo_angle = torch.cat([pseudo_sin, pseudo_cos], dim=-1)  # [num_edge, 2]
        embed_i = self.get_embedding(coff_i)  # [num_edge, C]
        embed_j = self.get_embedding(coff_j)  # [num_edge, C]
        edge_embed = torch.cat([pseudo_angle, embed_i, embed_j], dim=-1)
        edge_attr_3D_frame_invariant = self.project(edge_embed)
        
        edge_attr = edge_attr_2D + edge_attr_3D_frame_invariant
        
        # match dimension
        node_attr = self.node_emb(node_2D_repr)
        
        # estimate scores
        output = self.score_network(extended_edge_index, node_attr, edge_attr, equivariant_basis)
        output = output["gradient"]
        scores = -output

        _, std_pos = self.sde_pos.marGINal_prob(pos_perturbed, t_pos)
        scores = scores / std_pos[:, None]
        # print(t_pos, std_pos)
        return scores
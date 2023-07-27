import time
import os
import numpy as np
import torch
import torch.optim as optim

from tqdm import tqdm, trange
from torch_geometric.loader import DataLoader
from torch_geometric.nn import global_mean_pool
from torch_scatter import scatter

from Geom3D.datasets import Molecule3DDataset, MoleculeDataset3DRadius, MoleculeDatasetQM92D
from Geom3D.models import GNN, SchNet, PaiNN
from Geom3D.models.MoleculeSDE import SDEModel2Dto3D_01, SDEModel2Dto3D_02, SDEModel3Dto2D_node_adj_dense
from config import args
from torch_geometric.data import Data, Batch
import pickle
import abc

from Geom3D.models.MoleculeSDE.SDE_model_3D_to_2D_node_adj_dense import gen_noise, mask_adjs, mask_x, node_flags, to_dense_batch, to_dense_adj
from Geom3D.models.MoleculeSDE.SDE_sparse import VPSDE, VESDE, subVPSDE


def clip_norm(vec, limit, p=2):
    norm = torch.norm(vec, dim=-1, p=2, keepdim=True)
    denom = torch.where(norm > limit, limit / norm, torch.ones_like(norm))
    return vec * denom


def repeat_data(data, num_repeat) -> Batch:
    data_list = []
    for _ in range(num_repeat):
        neo_key_mapping = {}
        for key in data.keys:
            neo_key_mapping[key] = data[key]

        data_duplicate = Data.from_dict(neo_key_mapping)
        data_list.append(data_duplicate)
    return Batch.from_data_list(data_list)

def generate_samples_from_testset(data, molecule_model_2D, SDE_2Dto3D_model, args, out_path=None):
    molecule_model_2D.eval()
    SDE_2Dto3D_model.eval()
    start = args.start
    end =args.end
    eval_epoch = args.eval_epoch
    test_set = data

    all_data_list = []
    print("len of all data: %d" % len(test_set))
    print("data.num_graphs", data.num_graphs)

    for i in tqdm(range(data.num_graphs)):
        if i < start or i >= end:
            continue
        
        return_data = test_set[i].clone().detach()
        neo_data = test_set[i]
        num_repeat_ = args.num_repeat_SDE_inference
        repeated_data = repeat_data(neo_data, num_repeat_).to(device)
        
        # get node_3D_repr
        if args.model_3d == 'SchNet':
            _, node_3D_repr = molecule_model_3D(repeated_data.x[:, 0], repeated_data.positions, repeated_data.batch, return_latent=True)
        elif args.model_3d == "PaiNN":
            _, node_3D_repr = molecule_model_3D(repeated_data.x[:, 0], repeated_data.positions, repeated_data.radius_edge_index, repeated_data.batch, return_latent=True)

        # calculate max_num_nodes
        node2graph = repeated_data.batch
        B = node2graph.max().item() + 1
        one = node2graph.new_ones(node2graph.size(0))
        num_nodes = scatter(one, node2graph, dim=0, dim_size=B, reduce='add')
        max_num_nodes = num_nodes.max().item()

        num_class_X = SDE_3Dto2D_model.num_class_X
        
        node_3D_repr, _ = to_dense_batch(node_3D_repr, node2graph, max_num_nodes=max_num_nodes)  # [B, max_num_nodes, hdim]

        x, adj, x_mean, adj_mean = node_adj_PC_generation(
            representation=node_3D_repr, data=repeated_data,
            SDE_model=SDE_3Dto2D_model,
            B=B, max_num_nodes=max_num_nodes, num_class_X=num_class_X,
            n_steps=args.steps_pos,
        )
        repeated_data['x_SDE'] = x
        repeated_data['adj_SDE'] = adj

        # TODO:
        if i >= 9:
            break

    return all_data_list


@torch.no_grad()
def node_adj_PC_generation(
    representation, data, SDE_model,
    B, max_num_nodes, num_class_X,
    probability_flow=False,
    eps=1e-4, snr=0.2, scale_eps=0.9, n_steps=1,
):
    predictor_fn = ReverseDiffusionPredictor
    corrector_fn = LangevinCorrector

    sde_x = SDE_model.sde_x
    sde_adj = SDE_model.sde_adj

    score_fn_x = SDE_model.get_score_fn(
        sde_x, SDE_model.node_score_network, train=False, continuous=True)  # function with output dim [B, max_num_nodes, num_class_X] or [B, max_num_nodes, 1]
    score_fn_adj = SDE_model.get_score_fn(
        sde_adj, SDE_model.edge_score_network, train=False, continuous=True)

    predictor_obj_x = predictor_fn('x', sde_x, SDE_model, score_fn_x, probability_flow)
    corrector_obj_x = corrector_fn('x', sde_x, SDE_model, score_fn_x, snr, scale_eps, n_steps)

    predictor_obj_adj = predictor_fn('adj', sde_adj, SDE_model, score_fn_adj, probability_flow)
    corrector_obj_adj = corrector_fn('adj', sde_adj, SDE_model, score_fn_adj, snr, scale_eps, n_steps)

    # -------- Initial sample --------
    x_init = sde_x.prior_sampling((B, max_num_nodes, num_class_X)).to(device)
    adj_init = sde_adj.prior_sampling((B, max_num_nodes, max_num_nodes)).to(device)
    edge_attr = data.edge_attr[:, 0].float() + 1
    adj_oracle = to_dense_adj(data.edge_index, data.batch, edge_attr=edge_attr, max_num_nodes=max_num_nodes)
    flags = node_flags(adj_oracle)  # [B, max_num_nodes]
    x = x_init = mask_x(x_init, flags)  # [B, max_num_nodes, num_class_X]
    adj = adj_init = mask_adjs(adj_init, flags)  # [B, max_num_nodes, max_num_nodes]
    
    diff_steps = sde_adj.N
    timesteps = torch.linspace(sde_adj.T, eps, diff_steps, device=args.device)

    for i in trange(0, (diff_steps), desc='[Sampling]', position=1, leave=False):
        t = timesteps[i]
        vec_t = torch.ones(B, device=t.device) * t #(B,)

        _x = x  # [B, max_num_nodes, num_class_X]
        _adj = adj  # [B, max_num_nodes, max_num_nodes]
        adj, adj_mean = corrector_obj_adj.update_fn(representation, _x, _adj, flags, vec_t)  # [B, max_num_nodes, max_num_nodes], [B, max_num_nodes, max_num_nodes]
        x, x_mean = corrector_obj_x.update_fn(representation, _x, _adj, flags, vec_t)  # [B, max_num_nodes, num_class_X], [B, max_num_nodes, num_class_X]

        _x = x  # [B, max_num_nodes, num_class_X]
        _adj = adj  # [B, max_num_nodes, max_num_nodes]
        adj, adj_mean = predictor_obj_adj.update_fn(representation, _x, _adj, flags, vec_t)
        x, x_mean = predictor_obj_x.update_fn(representation, _x, _adj, flags, vec_t)

        # TODO: hacking
        if i >= 10:
            break
    print(' ')

    return x, adj, x_mean, adj_mean


class Predictor(abc.ABC):
    def __init__(self, sde, SDE_model, score_fn, probability_flow=False):
        super().__init__()
        self.sde = sde
        self.SDE_model = SDE_model
        # Compute the reverse SDE/ODE
        self.rsde = sde.reverse(score_fn, probability_flow)
        self.score_fn = score_fn

    @abc.abstractmethod
    def update_fn(self, representation, x, adj, flags, t):
        pass


class ReverseDiffusionPredictor(Predictor):
    def __init__(self, obj, sde, SDE_model, score_fn, probability_flow=False):
        super().__init__(sde, SDE_model, score_fn, probability_flow)
        self.obj = obj

    def update_fn(self, representation, x, adj, flags, t):
        SDE_model = self.SDE_model

        if self.obj == 'x':
            f, G = self.rsde.discretize(x, adj, flags, t, representation, SDE_model, is_adj=False)
            z = gen_noise(x, flags, sym=False)
            x_mean = x - f
            x = x_mean + G[:, None, None] * z
            return x, x_mean

        elif self.obj == 'adj':
            f, G = self.rsde.discretize(x, adj, flags, t, representation, SDE_model, is_adj=True)
            z = gen_noise(adj, flags)
            adj_mean = adj - f
            adj = adj_mean + G[:, None, None] * z
            return adj, adj_mean

        else:
            raise NotImplementedError(f"obj {self.obj} not yet supported.")


class Corrector(abc.ABC):
    def __init__(self, sde, SDE_model, score_fn, snr, scale_eps, n_steps):
        super().__init__()
        self.sde = sde
        self.SDE_model = SDE_model
        self.score_fn = score_fn
        self.snr = snr
        self.scale_eps = scale_eps
        self.n_steps = n_steps

    @abc.abstractmethod
    def update_fn(self, representation, x, adj, flags, t):
        pass


class LangevinCorrector(Corrector):
    def __init__(self, obj, sde, SDE_model, score_fn, snr, scale_eps, n_steps):
        super().__init__(sde, SDE_model, score_fn, snr, scale_eps, n_steps)
        self.obj = obj

    def update_fn(self, representation, x, adj, flags, t):
        sde = self.sde
        SDE_model = self.SDE_model
        score_fn = self.score_fn
        n_steps = self.n_steps
        target_snr = self.snr
        seps = self.scale_eps

        if isinstance(sde, VPSDE) or isinstance(sde, subVPSDE):
            timestep = (t * (sde.N - 1) / sde.T).long()
            alpha = sde.alphas.to(t.device)[timestep]
        else:
            alpha = torch.ones_like(t)

        if self.obj == 'x':
            perturbed_x = SDE_model.embedding_3D(representation) + SDE_model.embedding_X(x)  # [B, max_num_nodes, hdim]
            for _ in range(n_steps):
                grad = score_fn(perturbed_x, adj, flags, t)  # [B, max_num_nodes, num_class_X]
                noise = gen_noise(x, flags, sym=False)  # [B, max_num_nodes, num_class_X]
                grad_norm = torch.norm(grad.reshape(grad.shape[0], -1), dim=-1).mean()
                noise_norm = torch.norm(noise.reshape(noise.shape[0], -1), dim=-1).mean()
                step_size = (target_snr * noise_norm / grad_norm) ** 2 * 2 * alpha
                x_mean = x + step_size[:, None, None] * grad  # [B, max_num_nodes, hdim]
                x = x_mean + torch.sqrt(step_size * 2)[:, None, None] * noise * seps  # [B, max_num_nodes, hdim]
            return x, x_mean

        elif self.obj == 'adj':
            perturbed_x = SDE_model.embedding_3D(representation) + SDE_model.embedding_X(x)  # [B, max_num_nodes, hdim]
            for _ in range(n_steps):
                grad = score_fn(perturbed_x, adj, flags, t)  # [B, max_num_nodes, max_num_nodes]
                noise = gen_noise(adj, flags)  # [B, max_num_nodes, max_num_nodes]
                grad_norm = torch.norm(grad.reshape(grad.shape[0], -1), dim=-1).mean()
                noise_norm = torch.norm(noise.reshape(noise.shape[0], -1), dim=-1).mean()
                step_size = (target_snr * noise_norm / grad_norm) ** 2 * 2 * alpha
                adj_mean = adj + step_size[:, None, None] * grad  # [B, max_num_nodes, max_num_nodes]
                adj = adj_mean + torch.sqrt(step_size * 2)[:, None, None] * noise * seps  # [B, max_num_nodes, max_num_nodes]
            return adj, adj_mean

        else:
            raise NotImplementedError(f"obj {self.obj} not yet supported")



if __name__ == '__main__':
    torch.manual_seed(0)
    np.random.seed(0)
    device = torch.device('cuda:' + str(args.device)) if torch.cuda.is_available() else torch.device('cpu')
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(0)
        torch.cuda.set_device(args.device)
    node_class = 119

    # QM9 is only for debugGINg
    if args.dataset == "QM9":
        data_root = "{}/{}".format(args.input_data_dir, args.dataset)
        dataset = MoleculeDatasetQM92D(data_root, dataset=args.dataset, task=args.task)
    # This is only for preprocessing the dataset
    data_root = "{}/{}".format(args.input_data_dir, args.dataset)
    dataset = Molecule3DDataset(
        data_root, args.dataset, mask_ratio=args.SSL_masking_ratio, remove_center=True, use_extend_graph=args.use_extend_graph)
    if args.model_3d == "PaiNN":
        data_root = "{}_{}".format(data_root, args.PaiNN_radius_cutoff)
        dataset = MoleculeDataset3DRadius(
            data_root, preprcessed_dataset=dataset, radius=args.PaiNN_radius_cutoff, mask_ratio=args.SSL_masking_ratio, remove_center=True, use_extend_graph=args.use_extend_graph)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)

    # set up model
    molecule_model_2D = GNN(args.num_layer, args.emb_dim, JK=args.JK, drop_ratio=args.dropout_ratio, gnn_type=args.gnn_type).to(device)
    molecule_readout_func = global_mean_pool

    print('Using 3d model\t', args.model_3d)
    if args.model_3d == 'SchNet':
        molecule_model_3D = SchNet(
            hidden_channels=args.emb_dim,
            num_filters=args.num_filters,
            num_interactions=args.num_interactions,
            num_gaussians=args.num_gaussians,
            cutoff=args.cutoff,
            readout=args.readout,
            node_class=node_class,
        ).to(device)
    elif args.model_3d == "PaiNN":
        molecule_model_3D = PaiNN(
            n_atom_basis=args.emb_dim,
            n_interactions=args.PaiNN_n_interactions,
            n_rbf=args.PaiNN_n_rbf,
            cutoff=args.PaiNN_radius_cutoff,
            max_z=node_class,
            n_out=1,
            readout=args.PaiNN_readout,
        ).to(device)
    else:
        raise NotImplementedError('Model {} not included.'.format(args.model_3d))

    ###### same beta for VE and VP #####
    args.hidden_dim_2Dto3D = 32
    args.beta_schedule_2Dto3D = None
    if args.SDE_type_2Dto3D == "VE":
        args.beta_min_2Dto3D = 0.2
        args.beta_max_2Dto3D = 1.
        args.num_diffusion_timesteps_2Dto3D = 1000
    elif args.SDE_type_2Dto3D == "VP":
        args.beta_min_2Dto3D = 0.2
        args.beta_max_2Dto3D = 1.
        args.num_diffusion_timesteps_2Dto3D = 1000
        
    elif args.SDE_type_2Dto3D == "VE02":
        args.SDE_type_2Dto3D = "VE"
        args.beta_min_2Dto3D = 0.1
        args.beta_max_2Dto3D = 10.
        args.num_diffusion_timesteps_2Dto3D = 1000
    elif args.SDE_type_2Dto3D == "VP02":
        args.SDE_type_2Dto3D = "VP"
        args.beta_min_2Dto3D = 0.2
        args.beta_max_2Dto3D = 30.
        args.num_diffusion_timesteps_2Dto3D = 1000
        
    elif args.SDE_type_2Dto3D == "VE03":
        args.SDE_type_2Dto3D = "VE"
        args.beta_min_2Dto3D = 0.1
        args.beta_max_2Dto3D = 1000
        args.num_diffusion_timesteps_2Dto3D = 1000
    elif args.SDE_type_2Dto3D == "VP03":
        args.SDE_type_2Dto3D = "VP"
        args.beta_min_2Dto3D = 0.2
        args.beta_max_2Dto3D = 1000
        args.num_diffusion_timesteps_2Dto3D = 1000

    elif args.SDE_type_2Dto3D == "discrete_VE":
        args.beta_schedule_2Dto3D = "sigmoid"
        args.beta_min_2Dto3D = 1e-7
        args.beta_max_2Dto3D = 2e-3
        args.num_diffusion_timesteps_2Dto3D = 1000
        
    elif args.SDE_type_2Dto3D == "VE_test":
        args.beta_min_2Dto3D = 0.2
        args.beta_max_2Dto3D = 1.
        args.num_diffusion_timesteps_2Dto3D = 1000
    elif args.SDE_type_2Dto3D == "VP_test":
        args.beta_min_2Dto3D = 0.2
        args.beta_max_2Dto3D = 1.
        args.num_diffusion_timesteps_2Dto3D = 1000

    if args.SDE_2Dto3D_model == "SDEModel2Dto3D_01":
        SDE_2Dto3D_model = SDEModel2Dto3D_01(
                emb_dim=args.emb_dim, hidden_dim=args.hidden_dim_2Dto3D,
                beta_min=args.beta_min_2Dto3D, beta_max=args.beta_max_2Dto3D, num_diffusion_timesteps=args.num_diffusion_timesteps_2Dto3D,
                beta_schedule=args.beta_schedule_2Dto3D,
                SDE_type=args.SDE_type_2Dto3D,
                use_extend_graph=args.use_extend_graph).to(device)
    elif args.SDE_2Dto3D_model == "SDEModel2Dto3D_02":
        SDE_2Dto3D_model = SDEModel2Dto3D_02(
                emb_dim=args.emb_dim, hidden_dim=args.hidden_dim_2Dto3D,
                beta_min=args.beta_min_2Dto3D, beta_max=args.beta_max_2Dto3D, num_diffusion_timesteps=args.num_diffusion_timesteps_2Dto3D,
                beta_schedule=args.beta_schedule_2Dto3D,
                SDE_type=args.SDE_type_2Dto3D,
                use_extend_graph=args.use_extend_graph).to(device)

    node_class = 119
    ###### same beta for VE and VP #####
    if args.SDE_type_3Dto2D == "VE":
        args.beta_min_3Dto2D = 0.1
        args.beta_max_3Dto2D = 1.
        args.num_diffusion_timesteps_3Dto2D = 1000
    elif args.SDE_type_3Dto2D == "VP":
        args.beta_min_3Dto2D = 0.2
        args.beta_max_3Dto2D = 1.
        args.num_diffusion_timesteps_3Dto2D = 1000
    elif args.SDE_type_3Dto2D == "VE02":
        args.SDE_type_3Dto2D = "VE"
        args.beta_min_3Dto2D = 0.1
        args.beta_max_3Dto2D = 10.
        args.num_diffusion_timesteps_3Dto2D = 1000
    elif args.SDE_type_3Dto2D == "VP02":
        args.SDE_type_3Dto2D = "VP"
        args.beta_min_3Dto2D = 0.1
        args.beta_max_3Dto2D = 30.
        args.num_diffusion_timesteps_3Dto2D = 1000
    elif args.SDE_type_3Dto2D == "VE03":
        args.SDE_type_3Dto2D = "VE"
        args.beta_min_3Dto2D = 0.1
        args.beta_max_3Dto2D = 1000
        args.num_diffusion_timesteps_3Dto2D = 1000
    elif args.SDE_type_3Dto2D == "VP03":
        args.SDE_type_3Dto2D = "VP"
        args.beta_min_3Dto2D = 0.1
        args.beta_max_3Dto2D = 1000
        args.num_diffusion_timesteps_3Dto2D = 1000

    elif args.SDE_type_3Dto2D == "VE_test":
        args.SDE_type_3Dto2D = "VE"
        args.beta_min_3Dto2D = 0.1
        args.beta_max_3Dto2D = 1.
        args.num_diffusion_timesteps_3Dto2D = 1000
    elif args.SDE_type_3Dto2D == "VP_test":
        args.SDE_type_3Dto2D = "VP"
        args.beta_min_3Dto2D = 0.2
        args.beta_max_3Dto2D = 1.
        args.num_diffusion_timesteps_3Dto2D = 1000

    if args.noise_on_one_hot:
        reduce_mean = True
    else:
        reduce_mean = False
    if args.SDE_3Dto2D_model == "SDEModel3Dto2D_node_adj_dense":
        SDE_3Dto2D_model = SDEModel3Dto2D_node_adj_dense(
                dim3D=args.emb_dim, c_init=2, c_hid=8, c_final=4, num_heads=4, adim=16,
                nhid=16, num_layers=4,
                emb_dim=args.emb_dim, num_linears=3,
                beta_min=args.beta_min_3Dto2D, beta_max=args.beta_max_3Dto2D, num_diffusion_timesteps=args.num_diffusion_timesteps_3Dto2D,
                SDE_type=args.SDE_type_3Dto2D, num_class_X=node_class, noise_on_one_hot=args.noise_on_one_hot).to(device)

    for step, batch in enumerate(loader):
        batch = batch.to(device)
        node_2D_repr = molecule_model_2D(batch.x, batch.edge_index, batch.edge_attr)
        generate_samples_from_testset(batch, molecule_model_2D, SDE_2Dto3D_model, args, out_path="temp_inference_VE_VP")
        break

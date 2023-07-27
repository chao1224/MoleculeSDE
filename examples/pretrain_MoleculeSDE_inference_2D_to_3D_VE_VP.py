import time
import os
import numpy as np
import torch
import torch.optim as optim

from tqdm import tqdm, trange
from torch_geometric.loader import DataLoader
from torch_geometric.nn import global_mean_pool

from Geom3D.datasets import Molecule3DDataset, MoleculeDataset3DRadius, MoleculeDatasetQM92D
from Geom3D.models import GNN, SchNet, PaiNN
from Geom3D.models.MoleculeSDE import SDEModel2Dto3D_01, SDEModel2Dto3D_02, SDEModel3Dto2D_node_adj_dense
from util import dual_CL
from config import args
from torch_geometric.data import Data, Batch
import pickle
import abc

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
        neo_key_mapping["pos_gen"] = torch.ones_like(data.positions)

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
        num_repeat_ = args.num_repeat_SDE_inference
        batch = repeat_data(test_set[i], num_repeat_).to(args.device)
        representation = molecule_model_2D(batch.x, batch.edge_index, batch.edge_attr)

        pos_init = SDE_2Dto3D_model.sde_pos.prior_sampling(batch.positions.shape).to(args.device)
        data, pos_gen = position_PC_generation(
            representation=representation, data=batch, pos_init=pos_init, scorenet=SDE_2Dto3D_model,
            sde=SDE_2Dto3D_model.sde_pos, n_steps=args.steps_pos,
        )
        batch.pos_gen = pos_gen
       
        # TODO: stop here
        data_list = Batch.to_data_list(batch)

        all_pos = []
        for j in range(len(data_list)):
            all_pos.append(data_list[j].pos_gen)
        return_data.pos_gen = torch.cat(all_pos, 0)  # (num_repeat * num_node, 3)
        return_data.num_pos_gen = torch.tensor([len(all_pos)], dtype=torch.long)
        all_data_list.append(return_data)

        # TODO:
        if i >= 9:
            break

    if out_path is not None:
        output_path = os.path.join(out_path, "2D_to_3D_{}_{}_{}_epoch_{}_min_sig_{:.3f}_repeat_{}".format(args.generator, start, end, eval_epoch, args.min_sigma, args.num_repeat_SDE_inference),)
        with open(output_path, "wb") as fout:
            pickle.dump(all_data_list, fout)
        print("save generated %s samples to %s done!" % (args.generator, out_path))
    return all_data_list


@torch.no_grad()
def position_PC_generation(
        representation, data, pos_init, scorenet, sde, probability_flow=False, denoise=True,
        eps=1e-4, snr=0.2, scale_eps=0.9, n_steps=1,
    ):
    """
    # 1. initial pos: (N, 3)
    # 2. get d: (num_edge, 1)
    # 3. get score of d: score_d = self.get_grad(d).view(-1) (num_edge)
    # 4. get score of pos:
    #        dd_dr = (1/d) * (pos[edge_index[0]] - pos[edge_index[1]]) (num_edge, 3)
    #        edge2node = edge_index[0] (num_edge)
    #        score_pos = scatter_add(dd_dr * score_d, edge2node) (num_node, 3)
    # 5. update pos:
    #    pos = pos + step_size * score_pos + noise
    """
    predictor_fn = ReverseDiffusionPredictor
    corrector_fn = LangevinCorrector

    predictor_obj_x = predictor_fn(sde=sde, score_fn=scorenet, probability_flow=probability_flow)
    corrector_obj_x = corrector_fn(sde, scorenet, snr, scale_eps, n_steps)
    pos= pos_init
    with torch.no_grad():
        # -------- Initial sample --------

        diff_steps = scorenet.sde_pos.N
        timesteps = torch.linspace(scorenet.sde_pos.T, eps, diff_steps, device=args.device)
        for i in trange(0, (diff_steps), desc='[Sampling]', position=1, leave=False):
            t = timesteps[i]
            vec_t = torch.ones(data.num_graphs, device=t.device) * t #(num_graphs,)

            vec_t = vec_t.index_select(0, data.batch)  # (num_graph, )
            # corrector
            pos, pos_mean = corrector_obj_x.update_fn(representation, data, pos, vec_t, args)

            # predictor
            pos, pos_mean = predictor_obj_x.update_fn(representation, data, pos, vec_t, args)

            # TODO: hacking
            if i >= 10:
                break
        print(' ')

    if denoise:
        return data, pos_mean
    else:
        return data, pos


class Predictor(abc.ABC):
    """The abstract class for a predictor algorithm."""

    def __init__(self, sde, score_fn, probability_flow=False):
        super().__init__()
        self.sde = sde
        # Compute the reverse SDE/ODE
        self.rsde = sde.reverse(score_fn, probability_flow)
        self.score_fn = score_fn

    @abc.abstractmethod
    def update_fn(self, x, t, flags):
        pass


class ReverseDiffusionPredictor(Predictor):
    def __init__(self, sde, score_fn, probability_flow=False):
        super().__init__(sde, score_fn, probability_flow)
        # TODO: remove obj
        # self.obj = obj

    #representation, data,pos, vec_t,args
    def update_fn(self, representation, data, pos, t, args):
        f, G = self.rsde.discretize(pos, representation, data, t)
        noise = torch.randn_like(pos)
        x_mean = pos - f
        x = x_mean + G[:, None] * noise
        return x, x_mean


class Corrector(abc.ABC):
    """The abstract class for a corrector algorithm."""

    def __init__(self, sde, score_fn, snr, scale_eps, n_steps):
        super().__init__()
        self.sde = sde
        self.score_fn = score_fn
        self.snr = snr
        self.scale_eps = scale_eps
        self.n_steps = n_steps

    @abc.abstractmethod
    def update_fn(self, x, t, flags):
        pass


class LangevinCorrector(Corrector):
    def __init__(self, sde, score_fn, snr, scale_eps, n_steps):
        super().__init__(sde, score_fn, snr, scale_eps, n_steps)

    def update_fn(self, representation, data, pos, t, args):
        sde = self.sde
        score_fn = self.score_fn
        n_steps = self.n_steps
        target_snr = self.snr
        seps = self.scale_eps

        if isinstance(sde, VPSDE) or isinstance(sde, subVPSDE):
            timestep = (t * (sde.N - 1) / sde.T).long()
            alpha = sde.alphas.to(t.device)[timestep]
        else:
            alpha = torch.ones_like(t)

        for i in range(n_steps):
            grad = score_fn.get_score(representation, data, pos, None, t)
            noise = torch.randn_like(pos)
            grad_norm = torch.norm(grad.reshape(grad.shape[0], -1), dim=-1).mean()
            noise_norm = torch.norm(noise.reshape(noise.shape[0], -1), dim=-1).mean()
            step_size = (target_snr * noise_norm / grad_norm) ** 2 * 2 * alpha
            x_mean = pos + step_size[:, None] * grad
            x = x_mean + torch.sqrt(step_size * 2)[:, None] * noise * seps
        return x, x_mean


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

    '''
    python pretrain_MoleculeSDE.py --verbose --input_data_dir=temp --dataset=QM9
    python pretrain_MoleculeSDE.py --verbose --input_data_dir=../data --dataset=pcqm4mv2
    '''

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
    elif args.SDE_type_2Dto3D == "discrete_VE":
        args.beta_schedule_2Dto3D = "sigmoid"
        args.beta_min_2Dto3D = 1e-7
        args.beta_max_2Dto3D = 2e-3
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
        
    if args.SDE_3Dto2D_model == "SDEModel3Dto2D_node_adj_dense":
        SDE_3Dto2D_model = SDEModel3Dto2D_node_adj_dense(
                dim3D=args.emb_dim, c_init=2, c_hid=8, c_final=4, num_heads=4, adim=16,
                nhid=16, num_layers=4,
                emb_dim=args.emb_dim, num_linears=3,
                beta_min=args.beta_min_3Dto2D, beta_max=args.beta_max_3Dto2D, num_diffusion_timesteps=args.num_diffusion_timesteps_3Dto2D,
                SDE_type=args.SDE_type_3Dto2D, num_class_X=node_class, noise_on_one_hot=args.noise_on_one_hot).to(device)

    args.generator = "PC"

    for step, batch in enumerate(loader):
        batch = batch.to(device)
        node_2D_repr = molecule_model_2D(batch.x, batch.edge_index, batch.edge_attr)
        generate_samples_from_testset(batch, molecule_model_2D, SDE_2Dto3D_model, args, out_path="temp_inference_VE_VP")
        break

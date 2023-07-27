import time
import random
import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from tqdm import tqdm
from torch_geometric.loader import DataLoader
from torch_geometric.nn import global_mean_pool

from Geom3D.datasets import Molecule3DDataset, MoleculeDataset3DRadius
from Geom3D.models import GNN, SchNet, PaiNN
from Geom3D.models.MoleculeSDE import SDEModel2Dto3D_01, SDEModel2Dto3D_02, SDEModel3Dto2D_node_adj_dense, SDEModel3Dto2D_node_adj_dense_02, SDEModel3Dto2D_node_adj_dense_03
from util import dual_CL
from config import args


CE_criterion = nn.CrossEntropyLoss()
from ogb.utils.features import get_atom_feature_dims
ogb_feat_dim = get_atom_feature_dims()
ogb_feat_dim = [x - 1 for x in ogb_feat_dim]
ogb_feat_dim[-2] = 0
ogb_feat_dim[-1] = 0


def compute_accuracy(pred, target):
    return float(torch.sum(torch.max(pred.detach(), dim=1)[1] == target).cpu().item())/len(pred)


def do_2D_masking(batch, node_repr, molecule_atom_masking_model, masked_atom_indices):
    target = batch.x[masked_atom_indices][:, 0].detach()
    node_pred = molecule_atom_masking_model(node_repr[masked_atom_indices])
    loss = CE_criterion(node_pred, target)
    acc = compute_accuracy(node_pred, target)
    return loss, acc


def perturb(x, positions, mu, sigma):
    x_perturb = x

    device = positions.device
    positions_perturb = positions + torch.normal(mu, sigma, size=positions.size()).to(device)

    return x_perturb, positions_perturb


def do_3D_masking(args, batch, model, mu, sigma):
    positions = batch.positions

    x_01 = batch.x[:, 0]
    positions_01 = positions
    x_02, positions_02 = perturb(x_01, positions, mu, sigma)

    if args.model_3d == "SchNet":
        _, molecule_3D_repr_02 = model(x_02, positions_02, batch.batch, return_latent=True)
    elif args.model_3d == "PaiNN":
        _, molecule_3D_repr_02 = model(x_02, positions_02, batch.radius_edge_index, batch.batch, return_latent=True)

    super_edge_index = batch.super_edge_index

    u_pos_01 = torch.index_select(positions_01, dim=0, index=super_edge_index[0])
    v_pos_01 = torch.index_select(positions_01, dim=0, index=super_edge_index[1])
    distance_01 = torch.sqrt(torch.sum((u_pos_01-v_pos_01)**2, dim=1)).unsqueeze(1) # (num_edge, 1)

    loss = GeoSSL_moel(batch, molecule_3D_repr_02, distance_01)

    return loss


def save_model(save_best):
    if not args.output_model_dir == '':
        if save_best:
            global optimal_loss
            print('save model with loss: {:.5f}'.format(optimal_loss))
            output_model_path = os.path.join(args.output_model_dir, "model_complete.pth")
            saver_dict = {
                'model_2D': molecule_model_2D.state_dict(),
                'model_3D': molecule_model_3D.state_dict(),
                'SDE_2Dto3D_model': SDE_2Dto3D_model.state_dict(),
                'SDE_3Dto2D_model': SDE_3Dto2D_model.state_dict(),
            }
            # if args.SDE_coeff_2D_masking > 0:
            #     saver_dict['molecule_atom_masking_model'] = molecule_atom_masking_model.state_dict()
            # if args.SDE_coeff_3D_masking > 0:
            #     saver_dict['GeoSSL_moel'] = GeoSSL_moel.state_dict()
            torch.save(saver_dict, output_model_path)

        else:
            output_model_path = os.path.join(args.output_model_dir, "model_complete_final.pth")
            saver_dict = {
                'model_2D': molecule_model_2D.state_dict(),
                'model_3D': molecule_model_3D.state_dict(),
                'SDE_2Dto3D_model': SDE_2Dto3D_model.state_dict(),
                'SDE_3Dto2D_model': SDE_3Dto2D_model.state_dict(),
            }
            # if args.SDE_coeff_2D_masking > 0:
            #     saver_dict['molecule_atom_masking_model'] = molecule_atom_masking_model.state_dict()
            # if args.SDE_coeff_3D_masking > 0:
            #     saver_dict['GeoSSL_moel'] = GeoSSL_moel.state_dict()
            torch.save(saver_dict, output_model_path)
    return


def train(args, molecule_model_2D, device, loader, optimizer):
    start_time = time.time()

    molecule_model_2D.train()
    molecule_model_3D.train()
    SDE_2Dto3D_model.train()
    SDE_3Dto2D_model.train()
    # if args.SDE_coeff_2D_masking > 0:
    #     molecule_atom_masking_model.train()
    # if args.SDE_coeff_3D_masking > 0:
    #     GeoSSL_moel.train()

    SDE_loss_2Dto3D_accum, SDE_loss_3Dto2D_accum = 0, 0
    CL_loss_accum, CL_acc_accum = 0, 0

    if args.verbose:
        l = tqdm(loader)
    else:
        l = loader
    for step, batch in enumerate(l):
        batch = batch.to(device)

        node_2D_repr = molecule_model_2D(batch.x, batch.edge_index, batch.edge_attr)

        if args.model_3d == 'SchNet':
            _, node_3D_repr = molecule_model_3D(batch.x[:, 0], batch.positions, batch.batch, return_latent=True)
        elif args.model_3d == "PaiNN":
            _, node_3D_repr = molecule_model_3D(batch.x[:, 0], batch.positions, batch.radius_edge_index, batch.batch, return_latent=True)

        loss = 0
        if args.SDE_coeff_contrastive > 0:
            CL_loss, CL_acc = dual_CL(node_2D_repr, node_3D_repr, args)
            loss += CL_loss * args.SDE_coeff_contrastive
            CL_loss_accum += CL_loss.detach().cpu().item()
            CL_acc_accum += CL_acc

        if args.SDE_coeff_generative_2Dto3D > 0:
            SDE_loss_2Dto3D_result = SDE_2Dto3D_model(node_2D_repr, batch, anneal_power=args.SDE_anneal_power)
            SDE_loss_2Dto3D = SDE_loss_2Dto3D_result["position"]
            loss += SDE_loss_2Dto3D * args.SDE_coeff_generative_2Dto3D
            SDE_loss_2Dto3D_accum += SDE_loss_2Dto3D.detach().cpu().item()
        
        if args.SDE_coeff_generative_3Dto2D > 0:
            SDE_loss_3Dto2Dx, SDE_loss_3Dto2Dadj = SDE_3Dto2D_model(node_3D_repr, batch, reduce_mean=reduce_mean, continuous=True, train=True, anneal_power=args.SDE_anneal_power)
            SDE_loss_3Dto2D = (SDE_loss_3Dto2Dx + SDE_loss_3Dto2Dadj) * 0.5
            loss += SDE_loss_3Dto2D * args.SDE_coeff_generative_3Dto2D
            SDE_loss_3Dto2D_accum += SDE_loss_3Dto2D.detach().cpu().item()
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    global optimal_loss
    CL_loss_accum /= len(loader)
    CL_acc_accum /= len(loader)
    SDE_loss_2Dto3D_accum /= len(loader)
    SDE_loss_3Dto2D_accum /= len(loader)
    
    temp_loss = \
        args.SDE_coeff_contrastive * CL_loss_accum + \
        args.SDE_coeff_generative_2Dto3D * SDE_loss_2Dto3D_accum + \
        args.SDE_coeff_generative_3Dto2D * SDE_loss_3Dto2D_accum
    if temp_loss < optimal_loss:
        optimal_loss = temp_loss
        save_model(save_best=True)
        
    print('CL Loss: {:.5f}\tCL Acc: {:.5f}\t\tSDE 2Dto3D Loss: {:.5f}\tSDE 3Dto2D Loss: {:.5f}'.format(
        CL_loss_accum, CL_acc_accum, SDE_loss_2Dto3D_accum, SDE_loss_3Dto2D_accum))
    print('Time: {:.5f}\n'.format(time.time() - start_time))
    return


if __name__ == '__main__':
    torch.manual_seed(0)
    np.random.seed(0)
    device = torch.device('cuda:' + str(args.device)) if torch.cuda.is_available() else torch.device('cpu')
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(0)
        torch.cuda.set_device(args.device)
    node_class = 119

    transform = None
    data_root = "{}/{}".format(args.input_data_dir, args.dataset)
    dataset = Molecule3DDataset(
        data_root, args.dataset, mask_ratio=args.SSL_masking_ratio, remove_center=True, use_extend_graph=args.use_extend_graph, transform=transform)
    if args.model_3d == "PaiNN":
        data_root = "{}_{}".format(data_root, args.PaiNN_radius_cutoff)
        dataset = MoleculeDataset3DRadius(
            data_root, preprcessed_dataset=dataset, radius=args.PaiNN_radius_cutoff, mask_ratio=args.SSL_masking_ratio, remove_center=True, use_extend_graph=args.use_extend_graph)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)

    # set up model
    molecule_model_2D = GNN(args.num_layer, args.emb_dim, JK=args.JK, drop_ratio=args.dropout_ratio, gnn_type=args.gnn_type).to(device)
    molecule_readout_func = global_mean_pool

    print('Using 3d model\t', args.model_3d)
    if args.model_3d == "SchNet":
        molecule_model_3D = SchNet(
            hidden_channels=args.emb_dim,
            num_filters=args.SchNet_num_filters,
            num_interactions=args.SchNet_num_interactions,
            num_gaussians=args.SchNet_num_gaussians,
            cutoff=args.SchNet_cutoff,
            readout=args.SchNet_readout,
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
    elif args.SDE_3Dto2D_model == "SDEModel3Dto2D_node_adj_dense_02":
        SDE_3Dto2D_model = SDEModel3Dto2D_node_adj_dense_02(
                dim3D=args.emb_dim, c_init=2, c_hid=8, c_final=4, num_heads=4, adim=16,
                nhid=16, num_layers=4,
                emb_dim=args.emb_dim, num_linears=3,
                beta_min=args.beta_min_3Dto2D, beta_max=args.beta_max_3Dto2D, num_diffusion_timesteps=args.num_diffusion_timesteps_3Dto2D,
                SDE_type=args.SDE_type_3Dto2D, num_class_X=node_class, noise_on_one_hot=args.noise_on_one_hot).to(device)
    elif args.SDE_3Dto2D_model == "SDEModel3Dto2D_node_adj_dense_03":
        SDE_3Dto2D_model = SDEModel3Dto2D_node_adj_dense_03(
                dim3D=args.emb_dim, c_init=2, c_hid=8, c_final=4, num_heads=4, adim=16,
                nhid=16, num_layers=4,
                emb_dim=args.emb_dim, num_linears=3,
                beta_min=args.beta_min_3Dto2D, beta_max=args.beta_max_3Dto2D, num_diffusion_timesteps=args.num_diffusion_timesteps_3Dto2D,
                SDE_type=args.SDE_type_3Dto2D, num_class_X=node_class, noise_on_one_hot=args.noise_on_one_hot).to(device)

    model_param_group = []
    model_param_group.append({'params': molecule_model_2D.parameters(), 'lr': args.lr * args.gnn_2d_lr_scale})
    model_param_group.append({'params': molecule_model_3D.parameters(), 'lr': args.lr * args.gnn_3d_lr_scale})
    model_param_group.append({'params': SDE_2Dto3D_model.parameters(), 'lr': args.lr * args.gnn_2d_lr_scale})
    model_param_group.append({'params': SDE_3Dto2D_model.parameters(), 'lr': args.lr * args.gnn_3d_lr_scale})

    optimizer = optim.Adam(model_param_group, lr=args.lr, weight_decay=args.decay)
    optimal_loss = 1e10
    SDE_coeff_contrastive_oriGINal = args.SDE_coeff_contrastive
    args.SDE_coeff_contrastive = 0

    for epoch in range(1, args.epochs + 1):
        if epoch > args.SDE_coeff_contrastive_skip_epochs:
            args.SDE_coeff_contrastive = SDE_coeff_contrastive_oriGINal
        print('epoch: {}'.format(epoch))
        train(args, molecule_model_2D, device, loader, optimizer)

    save_model(save_best=False)

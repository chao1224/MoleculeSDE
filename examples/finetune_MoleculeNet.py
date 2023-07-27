import os
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch_geometric.loader import DataLoader
from torch_geometric.nn import global_max_pool, global_mean_pool
from tqdm import tqdm
from sklearn.metrics import roc_auc_score

from config import args
from Geom3D.datasets import MoleculeNetDataset2D
from Geom3D.models import GNN
from splitters import scaffold_split
from util import get_num_task


def mean_absolute_error(pred, target):
    return np.mean(np.abs(pred - target))


def preprocess_input(one_hot, charges, charge_power, charge_scale):
    charge_tensor = (charges.unsqueeze(-1) / charge_scale).pow(
        torch.arange(charge_power + 1.0, device=device, dtype=torch.float32)
    )  # (-1, 3)
    charge_tensor = charge_tensor.view(charges.shape + (1, charge_power + 1))
    atom_scalars = (
        one_hot.unsqueeze(-1) * charge_tensor
    )  # (N, charge_scale, charge_power + 1)
    atom_scalars = atom_scalars.view(
        charges.shape[:1] + (-1,)
    )  # (N, charge_scale * (charge_power + 1) )
    return atom_scalars


class RandomRotation(object):
    def __init__(self):
        pass

    def __call__(self, x):
        M = np.random.randn(3, 3)
        Q, __ = np.linalg.qr(M)
        return (x @ Q).float()


def split(dataset, smiles_list, data_root):
    train_dataset, valid_dataset, test_dataset = scaffold_split(
        dataset, smiles_list, null_value=0, frac_train=0.8, frac_valid=0.1, frac_test=0.1)
    print('split via scaffold')
    return train_dataset, valid_dataset, test_dataset


def model_setup():
    molecule_readout_func = None

    if args.model_2d == "GIN": # TODO: will fix this later
        model = GNN(args.num_layer, args.emb_dim, JK=args.JK, drop_ratio=args.dropout_ratio, gnn_type='GIN')
        molecule_readout_func = global_mean_pool
        graph_pred_linear = torch.nn.Linear(intermediate_dim, num_tasks)

    else:
        raise Exception("3D model {} not included.".format(args.model_2d))
    return model, molecule_readout_func, graph_pred_linear


def load_model(model, graph_pred_linear, model_weight_file):
    print("Loading from {}".format(model_weight_file))
    model_weight = torch.load(model_weight_file)  # , map_location='cpu'

    if "model_2D" in model_weight:
        model.load_state_dict(model_weight["model_2D"])
    elif "model" in model_weight:
        model.load_state_dict(model_weight["model"])
    else:
        model.load_state_dict(model_weight)
    return


def save_model(save_best):
    if not args.output_model_dir == "":
        if save_best:
            print("save model with optimal loss")
            output_model_path = os.path.join(args.output_model_dir, "model.pth")
            saved_model_dict = {}
            saved_model_dict["model"] = model.state_dict()
            if graph_pred_linear is not None:
                saved_model_dict["graph_pred_linear"] = graph_pred_linear.state_dict()
            torch.save(saved_model_dict, output_model_path)

        else:
            print("save model in the last epoch")
            output_model_path = os.path.join(args.output_model_dir, "model_final.pth")
            saved_model_dict = {}
            saved_model_dict["model"] = model.state_dict()
            if graph_pred_linear is not None:
                saved_model_dict["graph_pred_linear"] = graph_pred_linear.state_dict()
            torch.save(saved_model_dict, output_model_path)
    return


def train(epoch, device, loader, optimizer):
    model.train()
    if graph_pred_linear is not None:
        graph_pred_linear.train()

    loss_acc = 0
    num_iters = len(loader)

    if args.verbose:
        L = tqdm(loader)
    else:
        L = loader
    for step, batch in enumerate(L):
        batch = batch.to(device)

        if args.model_2d == "GIN":
            node_repr = model(batch.x, batch.edge_index, batch.edge_attr)
            molecule_2D_repr = molecule_readout_func(node_repr, batch.batch)

        if graph_pred_linear is not None:
            pred = graph_pred_linear(molecule_2D_repr)
        else:
            pred = molecule_2D_repr
        
        y = batch.y.view(pred.shape).to(torch.float64)

        # PaiNN can have some nan values
        mol_is_nan = torch.isnan(pred)
        if torch.sum(mol_is_nan) > 0:
            continue

        is_valid = y ** 2 > 0
        loss_mat = criterion(pred.double(), (y + 1) / 2)

        loss_mat = torch.where(is_valid, loss_mat, torch.zeros(loss_mat.shape).to(device).to(torch.float64))

        optimizer.zero_grad()
        loss = torch.sum(loss_mat) / torch.sum(is_valid)
        loss.backward()
        optimizer.step()
        loss_acc += loss.cpu().detach().item()

        if args.lr_scheduler in ["CosineAnnealingWarmRestarts"]:
            lr_scheduler.step(epoch - 1 + step / num_iters)

    loss_acc /= len(loader)
    if args.lr_scheduler in ["StepLR", "CosineAnnealingLR"]:
        lr_scheduler.step()
    elif args.lr_scheduler in [ "ReduceLROnPlateau"]:
        lr_scheduler.step(loss_acc)

    return loss_acc


@torch.no_grad()
def eval(device, loader):
    model.eval()
    if graph_pred_linear is not None:
        graph_pred_linear.eval()
    y_true, y_scores, y_valid = [], [], []

    if args.verbose:
        L = tqdm(loader)
    else:
        L = loader
    for batch in L:
        batch = batch.to(device)

        if args.model_2d == "GIN":
            node_repr = model(batch.x, batch.edge_index, batch.edge_attr)
            molecule_2D_repr = molecule_readout_func(node_repr, batch.batch)

        if graph_pred_linear is not None:
            pred = graph_pred_linear(molecule_2D_repr)
        else:
            pred = molecule_2D_repr
            
        true = batch.y.view(pred.shape)

        # PaiNN can have some nan values
        is_nan = torch.isnan(pred)
        # Whether y is non-null or not.
        is_valid = torch.logical_and((true ** 2 > 0), ~is_nan)

        y_true.append(true)
        y_scores.append(pred)
        y_valid.append(is_valid)

    y_true = torch.cat(y_true, dim=0).cpu().numpy()
    y_scores = torch.cat(y_scores, dim=0).cpu().numpy()
    y_valid = torch.cat(y_valid, dim=0).cpu().numpy()
    
    roc_list = []
    for i in range(y_true.shape[1]):
        try:
            is_valid = y_valid[:, i]
            roc_list.append(roc_auc_score((y_true[is_valid, i] + 1) / 2, y_scores[is_valid, i]))
        except:
            print('{} is invalid'.format(i))

    if len(roc_list) < y_true.shape[1]:
        print(len(roc_list))
        print('Some target is missing!')
        print('Missing ratio: %f' %(1 - float(len(roc_list)) / y_true.shape[1]))

    return sum(roc_list) / len(roc_list), y_true, y_scores


if __name__ == "__main__":
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = (
        torch.device("cuda:" + str(args.device))
        if torch.cuda.is_available()
        else torch.device("cpu")
    )
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    rotation_transform = None

    num_tasks = get_num_task(args.dataset)
    data_root_2D = "../data/molecule_datasets/{}".format(args.dataset)
    dataset = MoleculeNetDataset2D(data_root_2D, dataset=args.dataset)
    data_root = data_root_2D
    smiles_file = "{}/processed/smiles.csv".format(data_root)
    smiles_list = pd.read_csv(smiles_file, header=None)[0].tolist()
    
    train_dataset, valid_dataset, test_dataset = split(dataset, smiles_list, data_root)
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
    )
    val_loader = DataLoader(
        valid_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )
    criterion = nn.BCEWithLogitsLoss(reduction='none')

    # set up model
    if args.JK == "concat":
        intermediate_dim = (args.num_layer + 1) * args.emb_dim
    else:
        intermediate_dim = args.emb_dim

    node_class, edge_class = 119, 4
    model, molecule_readout_func, graph_pred_linear = model_setup()

    if args.input_model_file is not "":
        load_model(model, graph_pred_linear, args.input_model_file)
    model.to(device)
    print(model)
    if graph_pred_linear is not None:
        graph_pred_linear.to(device)
    print(graph_pred_linear)

    # set up optimizer
    # different learning rate for different part of GNN
    model_param_group = [{"params": model.parameters(), "lr": args.lr}]
    if graph_pred_linear is not None:
        model_param_group.append(
            {"params": graph_pred_linear.parameters(), "lr": args.lr}
        )
    optimizer = optim.Adam(model_param_group, lr=args.lr, weight_decay=args.decay)

    lr_scheduler = None
    if args.lr_scheduler == "CosineAnnealingLR":
        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, args.epochs
        )
        print("Apply lr scheduler CosineAnnealingLR")
    elif args.lr_scheduler == "CosineAnnealingWarmRestarts":
        lr_scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer, args.epochs, eta_min=1e-4
        )
        print("Apply lr scheduler CosineAnnealingWarmRestarts")
    elif args.lr_scheduler == "StepLR":
        lr_scheduler = optim.lr_scheduler.StepLR(
            optimizer, step_size=args.lr_decay_step_size, gamma=args.lr_decay_factor
        )
        print("Apply lr scheduler StepLR")
    elif args.lr_scheduler == "ReduceLROnPlateau":
        lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, factor=args.lr_decay_factor, patience=args.lr_decay_patience, min_lr=args.min_lr
        )
        print("Apply lr scheduler ReduceLROnPlateau")
    else:
        print("lr scheduler {} is not included.".format(args.lr_scheduler))

    train_roc_list, val_roc_list, test_roc_list = [], [], []
    best_val_roc, best_val_idx = 0, 0
    for epoch in range(1, args.epochs + 1):
        start_time = time.time()
        loss_acc = train(epoch, device, train_loader, optimizer)
        print("Epoch: {}\nLoss: {}".format(epoch, loss_acc))

        if epoch % args.print_every_epoch == 0:
            if args.eval_train:
                train_roc, train_target, train_pred = eval(device, train_loader)
            else:
                train_roc = 0
            val_roc, val_target, val_pred = eval(device, val_loader)
            test_roc, test_target, test_pred = eval(device, test_loader)

            train_roc_list.append(train_roc)
            val_roc_list.append(val_roc)
            test_roc_list.append(test_roc)
            print("train: {:.6f}\tval: {:.6f}\ttest: {:.6f}".format(train_roc, val_roc, test_roc))

            if val_roc > best_val_roc:
                best_val_roc = val_roc
                best_val_idx = len(train_roc_list) - 1
                if not args.output_model_dir == "":
                    save_model(save_best=True)

                    filename = os.path.join(
                        args.output_model_dir, "evaluation_best.pth"
                    )
                    np.savez(
                        filename,
                        val_target=val_target,
                        val_pred=val_pred,
                        test_target=test_target,
                        test_pred=test_pred,
                    )
        print("Took\t{}\n".format(time.time() - start_time))

    print("best train: {:.6f}\tval: {:.6f}\ttest: {:.6f}".format(train_roc_list[best_val_idx], val_roc_list[best_val_idx], test_roc_list[best_val_idx]))

    save_model(save_best=False)

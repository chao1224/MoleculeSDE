import argparse
from email.policy import default

parser = argparse.ArgumentParser()

# about seed and basic info
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--device", type=int, default=0)

parser.add_argument(
    "--model_3d",
    type=str,
    default="SchNet",
    choices=[
        "SchNet",
        "PaiNN",
    ],
)
parser.add_argument(
    "--model_2d",
    type=str,
    default="GIN",
    choices=[
        "GIN",
    ],
)

# about dataset and dataloader
parser.add_argument("--dataset", type=str, default="QM9")
parser.add_argument("--task", type=str, default="alpha")
parser.add_argument("--num_workers", type=int, default=0)

# for MD17
# The default hyper from here: https://github.com/divelab/DIG_storage/tree/main/3dgraph/MD17
parser.add_argument("--MD17_energy_coeff", type=float, default=0.05)
parser.add_argument("--MD17_force_coeff", type=float, default=0.95)
parser.add_argument("--energy_force_with_normalization", dest="energy_force_with_normalization", action="store_true")
parser.add_argument("--energy_force_no_normalization", dest="energy_force_with_normalization", action="store_false")
parser.set_defaults(energy_force_with_normalization=False)

# about training strategies
parser.add_argument("--split", type=str, default="customized_01",
                    choices=["customized_01", "customized_02", "random"])
parser.add_argument("--MD17_train_batch_size", type=int, default=1)
parser.add_argument("--batch_size", type=int, default=128)
parser.add_argument("--epochs", type=int, default=100)
parser.add_argument("--lr", type=float, default=1e-4)
parser.add_argument("--lr_scale", type=float, default=1)
parser.add_argument("--decay", type=float, default=0)
parser.add_argument("--print_every_epoch", type=int, default=1)
parser.add_argument("--loss", type=str, default="mae", choices=["mse", "mae"])
parser.add_argument("--lr_scheduler", type=str, default="CosineAnnealingLR")
parser.add_argument("--lr_decay_factor", type=float, default=0.5)
parser.add_argument("--lr_decay_step_size", type=int, default=100)
parser.add_argument("--lr_decay_patience", type=int, default=50)
parser.add_argument("--min_lr", type=float, default=1e-6)
parser.add_argument("--StepLRCustomized_scheduler", type=int, nargs='+', default=[150])
parser.add_argument("--verbose", dest="verbose", action="store_true")
parser.add_argument("--no_verbose", dest="verbose", action="store_false")
parser.set_defaults(verbose=False)
parser.add_argument("--use_rotation_transform", dest="use_rotation_transform", action="store_true")
parser.add_argument("--no_rotation_transform", dest="use_rotation_transform", action="store_false")
parser.set_defaults(use_rotation_transform=False)

# for SchNet
parser.add_argument("--SchNet_num_filters", type=int, default=128)
parser.add_argument("--SchNet_num_interactions", type=int, default=6)
parser.add_argument("--SchNet_num_gaussians", type=int, default=51)
parser.add_argument("--SchNet_cutoff", type=float, default=10)
parser.add_argument("--SchNet_readout", type=str, default="mean", choices=["mean", "add"])
parser.add_argument("--SchNet_gamma", type=float, default=None)

# for PaiNN
parser.add_argument("--PaiNN_radius_cutoff", type=float, default=5.0)
parser.add_argument("--PaiNN_n_interactions", type=int, default=3)
parser.add_argument("--PaiNN_n_rbf", type=int, default=20)
parser.add_argument("--PaiNN_readout", type=str, default="add", choices=["mean", "add"])
parser.add_argument("--PaiNN_gamma", type=float, default=None)

######################### for GraphMVP SSL #########################
### for 2D GNN
parser.add_argument("--gnn_type", type=str, default="GIN")
parser.add_argument("--num_layer", type=int, default=5)
parser.add_argument("--emb_dim", type=int, default=300)
parser.add_argument("--dropout_ratio", type=float, default=0.5)
parser.add_argument("--graph_pooling", type=str, default="mean")
parser.add_argument("--JK", type=str, default="last")
parser.add_argument("--gnn_2d_lr_scale", type=float, default=1)

######################### for GraphMVP SSL #########################
### for 3D GNN
parser.add_argument("--gnn_3d_lr_scale", type=float, default=1)

### for masking
parser.add_argument("--SSL_masking_ratio", type=float, default=0.15)

### for 2D-3D Contrastive SSL
parser.add_argument("--CL_neg_samples", type=int, default=1)
parser.add_argument("--CL_similarity_metric", type=str, default="InfoNCE_dot_prod",
                    choices=["InfoNCE_dot_prod", "EBM_dot_prod", "EBM_node_dot_prod"])
parser.add_argument("--T", type=float, default=0.1)
parser.add_argument("--normalize", dest="normalize", action="store_true")
parser.add_argument("--no_normalize", dest="normalize", action="store_false")
# parser.add_argument("--alpha_1", type=float, default=1)

### for MoleculeSDE
parser.add_argument("--SDE_type_2Dto3D", type=str, default="VE")
parser.add_argument("--SDE_type_3Dto2D", type=str, default="VE")
parser.add_argument("--SDE_2Dto3D_model", type=str, default="SDEModel2Dto3D_01")
parser.add_argument("--SDE_3Dto2D_model", type=str, default="SDEModel3Dto2D_node_adj_dense")
parser.add_argument("--SDE_coeff_contrastive", type=float, default=1)
parser.add_argument("--SDE_coeff_contrastive_skip_epochs", type=int, default=0)
parser.add_argument("--SDE_coeff_generative_2Dto3D", type=float, default=1)
parser.add_argument("--SDE_coeff_generative_3Dto2D", type=float, default=1)

# This is only for 3D to 2D
parser.add_argument("--use_extend_graph", dest="use_extend_graph", action="store_true")
parser.add_argument("--no_extend_graph", dest="use_extend_graph", action="store_false")
parser.set_defaults(use_extend_graph=True)
# This is only for 2D to 3D
parser.add_argument("--noise_on_one_hot", dest="noise_on_one_hot", action="store_true")
parser.add_argument("--no_noise_on_one_hot", dest="noise_on_one_hot", action="store_false")
parser.set_defaults(noise_on_one_hot=True)
parser.add_argument("--SDE_anneal_power", type=float, default=0)
# This is only for 2D to 3D to MoleculeNet property
parser.add_argument("--molecule_property_SDE_2D", type=float, default=1)

### for MoleculeSDE inference
parser.add_argument('--generator', type=str, help='type of generator [MultiScaleLD, PC]', default='MultiScaleLD')
parser.add_argument('--eval_epoch', type=int, default=None, help='evaluation epoch')
parser.add_argument('--start', type=int, default=0, help='start idx of test generation')
parser.add_argument('--end', type=int, default=100, help='end idx of test generation')
parser.add_argument('--num_repeat_SDE_inference', type=int, default=10, help='number of conformers')
parser.add_argument('--num_repeat_SDE_predict', type=int, default=1, help='number of conformers for prediction')
parser.add_argument("--min_sigma", type=float, default=0.0)
parser.add_argument('--steps_pos', type=int, default=100, help='MCMC')
parser.add_argument("--step_lr_pos", type=float, default=0.0000015)
parser.add_argument("--clip", type=float, default=1000)
parser.add_argument("--num_diffusion_timesteps_2Dto3D_inference", type=int, default=20)
parser.add_argument("--num_diffusion_timesteps_3Dto2D_inference", type=int, default=20)
parser.add_argument("--visualization_timesteps_interval", type=int, default=20)
parser.add_argument("--data_path_2D_SDE", type=str, default="")

parser.add_argument("--corrector_steps", type=int, default=1)

##### about if we would print out eval metric for training data
parser.add_argument("--eval_train", dest="eval_train", action="store_true")
parser.add_argument("--no_eval_train", dest="eval_train", action="store_false")
parser.set_defaults(eval_train=False)

parser.add_argument("--eval_test", dest="eval_test", action="store_true")
parser.add_argument("--no_eval_test", dest="eval_test", action="store_false")
parser.set_defaults(eval_test=True)

parser.add_argument("--input_data_dir", type=str, default="")

# about loading and saving
parser.add_argument("--input_model_file", type=str, default="")
parser.add_argument("--output_model_dir", type=str, default="")

parser.add_argument("--threshold", type=float, default=0)

args = parser.parse_args()
print("arguments\t", args)

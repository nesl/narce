import argparse
import numpy as np
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader
from torchinfo import summary

from utils import set_seeds, create_src_causal_mask, CEDataset, StateCEDataset
from train import test_narce, test, test_narce_iterative
from loss import focal_loss

from mamba_ssm.models.config_mamba import MambaConfig
from models import RNN, TCN, TSTransformer, BaselineMamba
from nar_model import NARMamba, AdapterMamba, NarcePipeline, StateNarce


parser = argparse.ArgumentParser(description='Model Evaluation')
group = parser.add_mutually_exclusive_group()
group.add_argument('--baseline', action='store_true')
group.add_argument('--narce', action='store_true')
parser.add_argument('-m', '--model', type=str, choices=['mamba2', 'narce_mamba2_6L', 'narce_mamba2_12L', 'state_narce_mamba2_12L'])
parser.add_argument('-s1', '--train_size', type=int, help='Size of the dataset this model is trained on', choices=[2000, 4000, 6000, 8000, 10000])
parser.add_argument('-s2', '--nar_train_size', type=int, help='Size of the dataset the NAR model is trained on', choices=[10000, 20000, 40000], required=False)
parser.add_argument('-d', '--dataset', type=str, help='Test dataset', choices=['5min-part', '5min-full', '15min-part', '15min-full'])
parser.add_argument('--seed', type=int, help='Random seed') #0, 17, 1243, 3674, 7341, 53, 97, 103, 191, 99719
args = parser.parse_args()

set_seeds(args.seed)


""" Setting """

batch_size = 256
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
criterion = focal_loss(alpha=torch.tensor([.005, 0.45, 0.45, 0.45]),gamma=2)
has_state = False
if args.model == 'state_narce_mamba2_12L':
    has_state = True


""" Load datasets """
if not has_state:
    if args.dataset == '5min-part':
        test_data_file = './data/CE_dataset/ce5min_test_data.npy'
        test_label_file = './data/CE_dataset/ce5min_test_labels.npy'
    elif args.dataset == '5min-full':
        test_data_file = './data/CE_dataset/ce5min_full_test_data.npy'
        test_label_file = './data/CE_dataset/ce5min_full_test_labels.npy'
    elif args.dataset == '15min-part':
        test_data_file = './data/CE_dataset/ce15min_test_data.npy'
        test_label_file = './data/CE_dataset/ce15min_test_labels.npy'
    elif args.dataset == '15min-full':
        test_data_file = './data/CE_dataset/ce15min_full_test_data.npy'
        test_label_file = './data/CE_dataset/ce15min_full_test_labels.npy'
    else:
        Exception("Dataset is not defined.") 
    test_data = np.load(test_data_file)
    test_labels = np.load(test_label_file)
    test_dataset = CEDataset(test_data, test_labels)

else:
    if args.dataset == '5min-part':
        test_data_file = './data/CE_dataset/state_ce5min_test.npz'
    elif args.dataset == '5min-full':
        test_data_file = './data/CE_dataset/state_ce5min_full_test.npz'
    elif args.dataset == '15min-part':
        test_data_file = './data/CE_dataset/state_ce15min_test.npz'
    elif args.dataset == '15min-full':
        test_data_file = './data/CE_dataset/state_ce15min_full_test.npz'
    else:
        Exception("Dataset is not defined.") 
    test_file = np.load(test_data_file)
    test_dataset = StateCEDataset(*test_file.values())

test_loader = DataLoader(test_dataset, 
                            batch_size=batch_size, 
                            shuffle=False
                            )

print(test_data_file)
print(test_loader.dataset.data.shape, test_loader.dataset.data.shape)


""" Load NN models """

input_dim = test_loader.dataset.data.shape[-1]
nar_vocab_size = 9 # Depends on the # unique tokens NAR takes in, which is # classes of atomic event
output_dim = 4 # The number of complex event classes

if args.baseline:
    if args.model == 'mamba2':
        mamba_config = MambaConfig(d_model=input_dim, n_layer=12, ssm_cfg={"layer": "Mamba2", "headdim": 32,})
        model = BaselineMamba(mamba_config, out_cls_dim=output_dim)

    else:
        raise Exception("Model is not defined.") 
    model_path = 'baseline/saved_model/{}-{}-{}.pt'.format(args.model, args.train_size, args.seed)
    
elif args.narce:
    if args.model == 'narce_mamba2_6L':
        nar_name= 'mamba2_v1'
        adapter_name = 'mamba2_6L'
        adapter_model = AdapterMamba(d_model=input_dim, n_layer=6)
    elif args.model == 'narce_mamba2_12L':
        nar_name= 'mamba2_v1'
        adapter_name = 'mamba2_12L'
        adapter_model = AdapterMamba(d_model=input_dim, n_layer=12)
    elif args.model == 'state_narce_mamba2_12L':
        nar_name= 'state_mamba2_v1'
        adapter_name = 'mamba2_12L'
        adapter_model = AdapterMamba(d_model=input_dim, n_layer=12)
    else:
        raise Exception("Model is not defined.") 
    
    # mamba2_v1
    mamba_config = MambaConfig(d_model=input_dim, n_layer=12, ssm_cfg={"layer": "Mamba2", "headdim": 32,})

    if not has_state:
        nar_model = NARMamba(mamba_config, nar_vocab_size=nar_vocab_size, out_cls_dim=output_dim).nar
        model = NarcePipeline(
            frozen_nar=nar_model,
            adapter_model=adapter_model,
        )
    else:
        state_vocab_size = 8 # The number of FSM states
        model = StateNarce(mamba_config, nar_vocab_size=nar_vocab_size, state_vocab_size=state_vocab_size, out_cls_dim=output_dim)
        model.in_encoder = adapter_model
    
    model_path = 'narce/saved_model/pipeline/{}-{}-{}-{}-{}.pt'.format(nar_name, args.nar_train_size, adapter_name, args.train_size, args.seed)


model.load_state_dict(torch.load(model_path))
summary(model)

Path('evaluate/baseline/plots/').mkdir(parents=True, exist_ok=True)
save_fig_dir = 'evaluate/baseline/plots/{}-{}-{}-{}.png'.format(args.dataset, args.model, args.train_size, args.seed)

""" Evaluation """

if not has_state:
    test(
        model=model,
        data_loader=test_loader,
        criterion=criterion,
        save_fig_dir=save_fig_dir,
        device=device
        )
else:
    test_narce_iterative(
        model=model,
        data_loader=test_loader,
        criterion=criterion,
        device=device
        )

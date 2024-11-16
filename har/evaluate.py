import argparse
import numpy as np
from pathlib import Path
import re

import torch
from torch import nn
from torch.utils.data import DataLoader
from torchinfo import summary

from utils import set_seeds, create_src_causal_mask, CEDataset
from train import test_narce, test, test_narce_iterative
from loss import focal_loss

from mamba_ssm.models.config_mamba import MambaConfig
from models import RNN, TCN, TSTransformer, BaselineMamba
from nar_model import NARMamba, AdapterMamba, NarcePipeline, StateNarce


parser = argparse.ArgumentParser(description='Model Evaluation')
group = parser.add_mutually_exclusive_group()
group.add_argument('--baseline', action='store_true')
group.add_argument('--narce', action='store_true')
parser.add_argument('-m', '--model', type=str, choices=['lstm', 'tcn', 'transformer', 'ae_lstm', 'ae_tcn', 'ae_transformer', \
                                                        'mamba1', 'mamba2', 'ae_mamba1', 'ae_mamba2', \
                                                        'soft_ae_lstm', 'soft_ae_tcn', 'soft_ae_transformer','soft_ae_mamba1'\
                                                        'narce_mamba1_2L','narce_mamba1_4L', 'narce_mamba1_6L','narce_mamba1_12L',\
                                                        'narce_mamba2_6L', 'narce_mamba2_12L', 'state_narce_mamba2_12L', 'narce_mlp_L'])
parser.add_argument('-s1', '--train_size', type=int, help='Size of the dataset this model is trained on', choices=[100, 200, 400, 500, 600, 800, 1000, 2000, 4000, 6000, 8000, 10000])
parser.add_argument('-s2', '--nar_train_size', type=int, help='Size of the dataset the NAR model is trained on', choices=[10000, 20000, 40000], required=False)
parser.add_argument('-d', '--dataset', type=str, help='Test dataset', choices=['3min', '5min', '15min', '30min'])
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
    if args.model == 'lstm' or args.model == 'tcn' or args.model == 'transformer' or args.model == 'mamba1' or args.model == 'mamba2' or args.narce is True:
        test_data_file = './data/CE_dataset/ce{}_test_data.npy'.format(args.dataset)
        test_label_file = './data/CE_dataset/ce{}_test_labels.npy'.format(args.dataset)
    elif args.model == 'ae_lstm' or args.model == 'ae_tcn' or args.model == 'ae_transformer' or args.model == 'ae_mamba1' or args.model == 'ae_mamba2':
        test_data_file = './data/CE_dataset/ae2ce{}_test_data.npy'.format(args.dataset)
        test_label_file = './data/CE_dataset/ae2ce{}_test_labels.npy'.format(args.dataset)
    elif args.model == 'soft_ae_lstm' or args.model == 'soft_ae_tcn' or args.model == 'soft_ae_transformer' or args.model == 'soft_ae_mamba1':
        test_data_file = './data/CE_dataset/softae2ce{}_test_data.npy'.format(args.dataset)
        test_label_file = './data/CE_dataset/softae2ce{}_test_labels.npy'.format(args.dataset)
    else:
        raise Exception("Undefined models.")
    
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
    if args.model == 'lstm' or args.model == 'ae_lstm' or args.model == 'soft_ae_lstm':
        model = RNN(input_dim=input_dim, hidden_dim=256, output_dim=output_dim, num_layer=5)

    elif args.model == 'tcn' or args.model == 'ae_tcn' or args.model == 'soft_ae_tcn':
        model = TCN(input_size=input_dim, output_size=output_dim, num_channels=[256,256,256,256,256,256], kernel_size=3, dropout=0.2)

    elif args.model == 'transformer' or args.model == 'ae_transformer' or args.model == 'soft_ae_transformer':
        model = TSTransformer(input_dim=input_dim, output_dim=output_dim, num_head=4, num_layers=6, pos_encoding=True)

    elif args.model == 'mamba1' or args.model == 'ae_mamba1' or args.model == 'soft_ae_mamba1':
        mamba_config = MambaConfig(d_model=128, n_layer=12, ssm_cfg={"layer": "Mamba1"})
        model = BaselineMamba(mamba_config, in_dim=input_dim, out_cls_dim=output_dim)

    elif args.model == 'mamba2' or args.model == 'ae_mamba2':
        mamba_config = MambaConfig(d_model=128, n_layer=12, ssm_cfg={"layer": "Mamba2", "headdim": 32,})
        model = BaselineMamba(mamba_config, in_dim=input_dim, out_cls_dim=output_dim)

    else:
        raise Exception("Model is not defined.") 
    model_path = 'baseline/saved_model/{}/{}-{}-{}.pt'.format(args.model, args.model, args.train_size, args.seed)
    
elif args.narce:
    if args.model == 'state_narce_mamba2_12L':
        nar_name= 'state_mamba2_v1'
        adapter_name = 'mamba2_12L'
        adapter_model = AdapterMamba(d_model=input_dim, n_layer=12)
        # mamba2_v1
        mamba_config = MambaConfig(d_model=input_dim, n_layer=12, ssm_cfg={"layer": "Mamba2", "headdim": 32,})
    else:
        match = re.match(r"narce_([a-zA-Z]+\d*)_(\d+)L", args.model)
        if match:
            adapter_model_type = match.group(1)  # Capture the model type (e.g., 'mamba1', 'mlp')
            num_layers = int(match.group(2))  # Capture the number of layers
        else:
            raise ValueError(f"Invalid model name format: {args.model}")
        
        nar_name = 'mamba1_v1'
        nar_mamba_config = MambaConfig(d_model=input_dim, n_layer=12, ssm_cfg={"layer": "Mamba1"})

        adapter_name = adapter_model_type + '_' + str(num_layers) + 'L'
        if adapter_model_type == 'mamba1':
            adapter_mamba_config = MambaConfig(d_model=input_dim, n_layer=num_layers, ssm_cfg={"layer": "Mamba1"})
            adapter_model = AdapterMamba(adapter_mamba_config)
        elif adapter_model_type == 'mlp':
            class MLPEncoder(nn.Module):
                def __init__(self, input_dim, hidden_dim, output_dim=128):
                    super(MLPEncoder, self).__init__()
                    self.fc1 = nn.Linear(input_dim, hidden_dim)
                    self.fc2 = nn.Linear(hidden_dim, hidden_dim)
                    self.fc3 = nn.Linear(hidden_dim, output_dim)
                    
                def forward(self, x):
                    x = nn.functional.relu(self.fc1(x))
                    x = nn.functional.relu(self.fc2(x))
                    x = self.fc3(x)
                    return x

            
            adapter_model = MLPEncoder(input_dim=input_dim, hidden_dim=256)
            # adapter = MLPEncoder(input_dim=embedding_input_dim)
        else:
            raise Exception("Model is not defined.") 

    if not has_state:
        nar_model = NARMamba(nar_mamba_config, nar_vocab_size=nar_vocab_size, out_cls_dim=output_dim).nar
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

if args.baseline:
    Path('evaluate/baseline/plots/{}/'.format(args.model)).mkdir(parents=True, exist_ok=True)
    save_fig_dir = 'evaluate/baseline/plots/{}/{}-{}-{}-{}.png'.format(args.model, args.dataset, args.model, args.train_size, args.seed)
elif args.narce:
    Path('evaluate/narce/plots/{}/'.format(args.model)).mkdir(parents=True, exist_ok=True)
    save_fig_dir = 'evaluate/narce/plots/{}/{}-{}-{}-{}.png'.format(args.model, args.dataset, args.model, args.train_size, args.seed)
else:
    raise Exception("Experiment model type is not defined.") 


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

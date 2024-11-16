import argparse
import numpy as np
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader
from torchinfo import summary

from utils import set_seeds, create_src_causal_mask, CEDataset
from train import train, test
from loss import focal_loss

from models import RNN, TCN, TSTransformer, BaselineMamba

from mamba_ssm.models.config_mamba import MambaConfig

 
parser = argparse.ArgumentParser(description='NN Model Evaluation')
parser.add_argument('model', type=str, choices=['lstm', 'tcn', 'transformer', 'ae_lstm', 'ae_tcn', 'ae_transformer',  \
                                                'mamba1', 'mamba2', 'ae_mamba1', 'ae_mamba2'\
                                                'soft_ae_lstm', 'soft_ae_tcn', 'soft_ae_transformer','soft_ae_mamba1'])
parser.add_argument('dataset', type=int, help='Dataset size', choices=[100, 200, 400, 500, 600, 800, 1000, 2000, 4000, 6000, 8000, 10000, 20000])
parser.add_argument('seed',  type=int, help='Random seed') #0, 17, 1243, 3674, 7341, 53, 97, 103, 191, 99719

args = parser.parse_args()

set_seeds(args.seed)


""" Setting """

batch_size = 256
n_epochs = 5000
learning_rate = 1e-3
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
criterion = focal_loss(alpha=torch.tensor([.005, 0.45, 0.45, 0.45]),gamma=2)


""" Load datasets """

if args.model == 'lstm' or args.model == 'tcn' or args.model == 'transformer' or args.model == 'mamba1' or args.model == 'mamba2':
    train_data_file = './data/CE_dataset/ce5min_train_data_{}.npy'.format(args.dataset)
    train_label_file = './data/CE_dataset/ce5min_train_labels_{}.npy'.format(args.dataset)
    val_data_file = './data/CE_dataset/ce5min_val_data.npy'
    val_label_file = './data/CE_dataset/ce5min_val_labels.npy'
    test_data_file = './data/CE_dataset/ce5min_test_data.npy'
    test_label_file = './data/CE_dataset/ce5min_test_labels.npy'

elif args.model == 'ae_lstm' or args.model == 'ae_tcn' or args.model == 'ae_transformer' or args.model == 'ae_mamba1' or args.model == 'ae_mamba2':
    train_data_file = './data/CE_dataset/ae2ce5min_train_data_{}.npy'.format(args.dataset)
    train_label_file = './data/CE_dataset/ae2ce5min_train_labels_{}.npy'.format(args.dataset)
    val_data_file = './data/CE_dataset/ae2ce5min_val_data.npy'
    val_label_file = './data/CE_dataset/ae2ce5min_val_labels.npy'
    test_data_file = './data/CE_dataset/ae2ce5min_test_data.npy'
    test_label_file = './data/CE_dataset/ae2ce5min_test_labels.npy'

elif args.model == 'soft_ae_lstm' or args.model == 'soft_ae_tcn' or args.model == 'soft_ae_transformer' or args.model == 'soft_ae_mamba1':
    train_data_file = './data/CE_dataset/softae2ce5min_train_data_{}.npy'.format(args.dataset)
    train_label_file = './data/CE_dataset/softae2ce5min_train_labels_{}.npy'.format(args.dataset)
    val_data_file = './data/CE_dataset/softae2ce5min_val_data.npy'
    val_label_file = './data/CE_dataset/softae2ce5min_val_labels.npy'
    test_data_file = './data/CE_dataset/softae2ce5min_test_data.npy'
    test_label_file = './data/CE_dataset/softae2ce5min_test_labels.npy'

else:
    raise Exception("Unknown dataset.")

ce_train_data = np.load(train_data_file)
ce_train_labels = np.load(train_label_file)
ce_val_data = np.load(val_data_file)
ce_val_labels = np.load(val_label_file)
ce_test_data = np.load(test_data_file)
ce_test_labels = np.load(test_label_file)

print(train_data_file)
print(ce_train_data.shape, ce_train_labels.shape, ce_val_data.shape, ce_val_labels.shape, ce_test_data.shape, ce_test_labels.shape)

ce_train_dataset = CEDataset(ce_train_data, ce_train_labels)
ce_train_loader = DataLoader(ce_train_dataset, 
                             batch_size=batch_size, 
                             shuffle=True, 
                             num_workers=2
                             )
ce_val_dataset = CEDataset(ce_val_data, ce_val_labels)
ce_val_loader = DataLoader(ce_val_dataset,
                            batch_size=batch_size,
                            shuffle=False, 
                            )
ce_test_dataset = CEDataset(ce_test_data, ce_test_labels)
ce_test_loader = DataLoader(ce_test_dataset, 
                            batch_size=batch_size, 
                            shuffle=False
                            )


""" Load NN models """

input_dim = ce_train_data.shape[-1]
output_dim = 4

if args.model == 'lstm' or args.model == 'ae_lstm' or args.model == 'soft_ae_lstm':
    earlystop_min_delta = 1e-2
    model = RNN(input_dim=input_dim, hidden_dim=256, output_dim=output_dim, num_layer=5)

elif args.model == 'tcn' or args.model == 'ae_tcn' or args.model == 'soft_ae_tcn':
    earlystop_min_delta = 1e-2
    model = TCN(input_size=input_dim, output_size=output_dim, num_channels=[256,256,256,256,256,256], kernel_size=3, dropout=0.2)

elif args.model == 'transformer' or args.model == 'ae_transformer' or args.model == 'soft_ae_transformer':
    earlystop_min_delta = 1e-2
    model = TSTransformer(input_dim=input_dim, output_dim=output_dim, num_head=4, num_layers=6, pos_encoding=True)

elif args.model == 'mamba1' or args.model == 'ae_mamba1' or args.model == 'soft_ae_mamba1':
    earlystop_min_delta = 1e-3
    mamba_config = MambaConfig(d_model=128, n_layer=12, ssm_cfg={"layer": "Mamba1"})
    model = BaselineMamba(mamba_config, in_dim=input_dim, out_cls_dim=output_dim)

elif args.model == 'mamba2' or args.model == 'ae_mamba2':
    earlystop_min_delta = 1e-4
    mamba_config = MambaConfig(d_model=128, n_layer=12, ssm_cfg={"layer": "Mamba2", "headdim": 32,})
    model = BaselineMamba(mamba_config, in_dim=input_dim, out_cls_dim=output_dim)

else:
    raise Exception("Model is not defined.") 

summary(model)


""" Training and Testing """

# Creare dirctory if it doesn't exist
Path('baseline/saved_model/{}/'.format(args.model)).mkdir(parents=True, exist_ok=True)
model_path = 'baseline/saved_model/{}/{}-{}-{}.pt'.format(args.model, args.model, args.dataset, args.seed)
Path('baseline/plots/{}/'.format(args.model)).mkdir(parents=True, exist_ok=True)
save_fig_dir = 'baseline/plots/{}/{}-{}-{}.png'.format(args.model, args.model, args.dataset, args.seed)

src_causal_mask = create_src_causal_mask(ce_train_data.shape[1]) if args.model == 'transformer' or args.model == 'ae_transformer' else None

train(
    model=model,
    train_loader=ce_train_loader,
    val_loader=ce_val_loader, 
    n_epochs=n_epochs,
    lr=learning_rate,
    criterion=criterion,
    min_delta=earlystop_min_delta,
    save_path=model_path,
    src_mask=src_causal_mask,
    device=device
    )

# model.load_state_dict(torch.load(model_path))

test(
    model=model,
    data_loader=ce_test_loader,
    criterion=criterion,
    save_fig_dir=save_fig_dir,
    src_mask=src_causal_mask,
    device=device
    )
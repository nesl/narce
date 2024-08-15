import argparse
import numpy as np
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader
from torchinfo import summary

from utils import set_seeds, CEDataset
from train import train_narce, test_narce
from loss import focal_loss

from mamba_ssm.models.config_mamba import MambaConfig
from nar_model import NARMamba, AdapterMamba, NarcePipeline

import os
import wandb
os.environ['WANDB_API_KEY'] = '61020f05f6bfee65de753cef7157e59124d0145f'
 
parser = argparse.ArgumentParser(description='NARCE Pipeline Training')
parser.add_argument('nar_model', type=str, choices=['mamba2_v1'])
parser.add_argument('adapter_model', type=str, choices=['mamba2_6L', 'mamba2_12L'])
parser.add_argument('train', type=str, choices=['pipeline', 'nar', 'adapter'])
parser.add_argument('nar_dataset', type=int, help='Dataset size', choices=[10000, 20000, 40000])
parser.add_argument('sensor_dataset', type=int, help='Dataset size', choices=[2000, 4000, 6000, 8000, 10000])
parser.add_argument('seed',  type=int, help='Random seed') #0, 17, 1243, 3674, 7341, 53, 97, 103, 191, 99719
parser.add_argument('-g', '--gpu', default=0, type=int, choices=[0,1,2,3], required=False)

args = parser.parse_args()

set_seeds(args.seed)


""" Setting """

batch_size_nar = 256
batch_size_adapter = 256
n_epochs_nar = 5000
n_epochs_adapter = 20000
learning_rate_nar = 1e-3
learning_rate_adapter = 1e-3
device = torch.device("cuda:{}".format(args.gpu) if torch.cuda.is_available() else "cpu")
criterion = focal_loss(alpha=torch.tensor([.005, 0.45, 0.45, 0.45]),gamma=2)


""" Load datasets """
nar_train_data_file = './data/CE_dataset/nar_5min_full_train_data_{}.npy'.format(args.nar_dataset)
nar_train_label_file = './data/CE_dataset/nar_5min_full_train_labels_{}.npy'.format(args.nar_dataset)
nar_val_data_file = './data/CE_dataset/nar_5min_full_val_data.npy'
nar_val_label_file = './data/CE_dataset/nar_5min_full_val_labels.npy'
nar_test_data_file = './data/CE_dataset/nar_5min_full_test_data.npy'
nar_test_label_file = './data/CE_dataset/nar_5min_full_test_labels.npy'

# sensor_train_data_file = './data/CE_dataset/ce5min_train_data_{}.npy'.format(args.sensor_dataset)
# sensor_train_label_file = './data/CE_dataset/ce5min_train_labels_{}.npy'.format(args.sensor_dataset)
# sensor_val_data_file = './data/CE_dataset/ce5min_val_data.npy'
# sensor_val_label_file = './data/CE_dataset/ce5min_val_labels.npy'
# sensor_test_data_file = './data/CE_dataset/ce5min_test_data.npy'
# sensor_test_label_file = './data/CE_dataset/ce5min_test_labels.npy'

sensor_train_data_file = './data/CE_dataset/ce5min_full_train_data_{}.npy'.format(args.sensor_dataset)
sensor_train_label_file = './data/CE_dataset/ce5min_full_train_labels_{}.npy'.format(args.sensor_dataset)
sensor_val_data_file = './data/CE_dataset/ce5min_full_val_data.npy'
sensor_val_label_file = './data/CE_dataset/ce5min_full_val_labels.npy'
sensor_test_data_file = './data/CE_dataset/ce5min_full_test_data.npy'
sensor_test_label_file = './data/CE_dataset/ce5min_full_test_labels.npy'

nar_train_data = np.load(nar_train_data_file)
nar_train_labels = np.load(nar_train_label_file)
nar_val_data = np.load(nar_val_data_file)
nar_val_labels = np.load(nar_val_label_file)
nar_test_data = np.load(nar_test_data_file)
nar_test_labels = np.load(nar_test_label_file)

sensor_train_data = np.load(sensor_train_data_file)
sensor_train_labels = np.load(sensor_train_label_file)
sensor_val_data = np.load(sensor_val_data_file)
sensor_val_labels = np.load(sensor_val_label_file)
sensor_test_data = np.load(sensor_test_data_file)
sensor_test_labels = np.load(sensor_test_label_file)

print(nar_train_data_file)
print(nar_train_data.shape, nar_train_labels.shape, nar_val_data.shape, nar_val_labels.shape, nar_test_data.shape, nar_test_labels.shape)
print(sensor_train_data_file)
print(sensor_train_data.shape, sensor_train_labels.shape, sensor_val_data.shape, sensor_val_labels.shape, sensor_test_data.shape, sensor_test_labels.shape)

nar_train_dataset = CEDataset(nar_train_data, nar_train_labels)
nar_train_loader = DataLoader(nar_train_dataset, 
                             batch_size=batch_size_nar, 
                             shuffle=True, 
                            #  num_workers=4
                             )
nar_val_dataset = CEDataset(nar_val_data, nar_val_labels)
nar_val_loader = DataLoader(nar_val_dataset,
                            batch_size=batch_size_nar,
                            shuffle=False, 
                            )
nar_test_dataset = CEDataset(nar_test_data, nar_test_labels)
nar_test_loader = DataLoader(nar_test_dataset, 
                            batch_size=batch_size_nar, 
                            shuffle=False
                            )

sensor_train_dataset = CEDataset(sensor_train_data, sensor_train_labels)
sensor_train_loader = DataLoader(sensor_train_dataset, 
                                batch_size=batch_size_adapter, 
                                shuffle=True, 
                                #  num_workers=4
                                )
sensor_val_dataset = CEDataset(sensor_val_data, sensor_val_labels)
sensor_val_loader = DataLoader(sensor_val_dataset, 
                                batch_size=batch_size_adapter, 
                                shuffle=False
                                )
sensor_test_dataset = CEDataset(sensor_test_data, sensor_test_labels)
sensor_test_loader = DataLoader(sensor_test_dataset, 
                                batch_size=batch_size_adapter, 
                                shuffle=False
                                )


""" Load NN models """

embedding_input_dim = sensor_train_data.shape[-1]
nar_vocab_size = 9 # Depends on the # unique tokens NAR takes in, which is # classes of atomic event
output_dim = 4 # The number of complex event classes

if args.nar_model == 'mamba2_v1':
    mamba_config = MambaConfig(d_model=embedding_input_dim, n_layer=12, ssm_cfg={"layer": "Mamba2", "headdim": 32,})
    embed_nar_model = NARMamba(mamba_config, nar_vocab_size=nar_vocab_size, out_cls_dim=output_dim)

else:
    raise Exception("Model is not defined.") 

summary(embed_nar_model)


""" Training and Testing """

# Creare dirctory if it doesn't exist
Path('narce/saved_model/nar/').mkdir(parents=True, exist_ok=True)
embed_nar_model_path = 'narce/saved_model/nar/{}-{}-{}.pt'.format(args.nar_model, args.nar_dataset, args.seed)
# embed_nar_model_path = 'narce/saved_model/nar/{}-{}-53.pt'.format(args.nar_model, args.nar_dataset) # Use the current best model

if args.train == 'pipeline' or args.train == 'nar':
    # First phase: training NAR
    
    run = wandb.init(
    # Set the project where this run will be logged
    project="narce-project",
    name="train-nar-falcon-{}".format(args.seed),
    # Track hyperparameters and run metadata
    config={
        "nar_learning_rate": learning_rate_nar,
        "nar_epochs": n_epochs_nar,
        "nar_batch_size": batch_size_nar,
        "nar_train_data": nar_train_data_file,
        "model": embed_nar_model_path,
        },
    )

    train_narce(
        model=embed_nar_model,
        train_loader=nar_train_loader,
        val_loader=nar_val_loader, # need to generate validation data
        n_epochs=n_epochs_nar,
        lr=learning_rate_nar,
        criterion=criterion,
        save_path=embed_nar_model_path,
        device=device
        )
    
    wandb.finish()

    test_narce(
        model=embed_nar_model,
        data_loader=nar_test_loader,
        criterion=criterion,
        device=device
        )
else:
    # Or, skip Phase 1 and load pretrained NAR
    embed_nar_model.load_state_dict(torch.load(embed_nar_model_path))


if args.train == 'pipeline' or args.train == 'adapter':
    # Second phase: training encoder while keeping NAR frozen

    nar = embed_nar_model.nar
    for param in nar.parameters():
        param.requires_grad = False

    if args.adapter_model == 'mamba2_6L':
        n_layer = 6
    elif args.adapter_model == 'mamba2_12L':
        n_layer = 12
    else:
        raise Exception("Model is not defined.") 

    adapter = AdapterMamba(d_model=embedding_input_dim, n_layer=n_layer)

    narce_model = NarcePipeline(
        frozen_nar=nar,
        adapter_model=adapter,
    )


    print("======================================================================")
    print("Check frozen params.")
    params = sum(p.numel() for p in narce_model.parameters())
    print("# Total parameters:", params)
    nar_params = sum(p.numel() for p in narce_model.nar.parameters())
    print("# NAR parameters:", nar_params)
    frozen_params = sum(p.numel() for p in narce_model.parameters() if p.requires_grad is False)
    print("# frozen parameters:", frozen_params)
    print("======================================================================")

    summary(narce_model)

    Path('narce/saved_model/pipeline/').mkdir(parents=True, exist_ok=True)
    narce_model_path = 'narce/saved_model/test/{}-{}-{}-{}-{}.pt'.format(args.nar_model, args.nar_dataset, args.adapter_model, args.sensor_dataset, args.seed)

    run = wandb.init(
    # Set the project where this run will be logged
    project="narce-project",
    name="train-adapter-falcon-{}".format(args.seed),
    # Track hyperparameters and run metadata
    config={
        "adapter_learning_rate": learning_rate_adapter,
        "adapter_epochs": n_epochs_adapter,
        "adapter_batch_size": batch_size_adapter,
        "adapter_train_data": sensor_train_data_file,
        "model": narce_model_path,
        },
    )

    train_narce(
        model=narce_model,
        train_loader=sensor_train_loader,
        val_loader=sensor_val_loader, # need to generate validation data
        n_epochs=n_epochs_adapter,
        lr=learning_rate_adapter,
        criterion=criterion,
        save_path=narce_model_path,
        is_train_adapter=True,
        device=device
        )
    
    wandb.finish()

    test_narce(
        model=narce_model,
        data_loader=sensor_test_loader,
        criterion=criterion,
        device=device
        )

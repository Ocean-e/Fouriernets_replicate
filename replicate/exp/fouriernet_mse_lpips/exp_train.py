# fouriernet:train
# loss_function: mse+lpips

import os
import datetime
import time
import logging
import subprocess
import math
import numpy as np
import pandas as pd
import sys

import torch
import torch.cuda.comm
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.cuda._utils import _get_device_index
from collections import OrderedDict

from ...utils.control import *
from ...utils.output_control import *
from ...utils.networks import *
from ...utils.dataset import DiffuserMirflickrDataset

# check and set GPU
if torch.cuda.is_available():
    torch.cuda.set_device(5)  #TODO:change GPU device
    device = torch.cuda.current_device()
    print("Current GPU device: ", device)
    device_count = torch.cuda.device_count()
    print(f"Number of available GPUs: {device_count}")
else:
    print("No CUDA devices available")


logging.basicConfig(filename="out.log", level=logging.DEBUG, format="%(message)s")

# define optimization hyperparameters
learning_rate = 1e-4
lpips_weight = 0
lpips_step_size = 0.1
lpips_step_milestones = [3000,6000,9000,9500]
num_iterations = 10000

# setup image shape and devices
image_shape = (1080, 1920)
devices = [torch.cuda.current_device()]

# calculate downsampled sizes
downsample = 4
downsampled_image_shape = [int(s / downsample) for s in image_shape]

def create_reconstruction_network():
    deconv = FourierNetRGB(
        20, 
        downsampled_image_shape,
        fourier_conv_args={"stride": 2},
        conv_kernel_sizes=[(11, 11), (11, 11), (11, 11)],
        conv_fmap_nums=[64, 64, 3],
        input_scaling_mode=None,
        device=devices[0],
    )
    return deconv

def initialize_reconstruction(latest=None):
    deconv = create_reconstruction_network()
    if latest is not None:
        print("[info] loading from checkpoint")
        deconv.load_state_dict(latest["deconv_state_dict"], strict=True) 
    return deconv

def initialize_optimizer(deconv, latest=None):
    opt = optim.Adam(
        [{"params": deconv.parameters(), "lr": learning_rate}], lr=learning_rate
    ) 
    if latest is not None:
        opt.load_state_dict(latest["opt_state_dict"])
    return opt

def create_dataset(test=False):
    # TODO:change base_path
    base_path = "/home/lihaiyue/data/snapshotscope/data/dlmd/dataset"  
    data_dir = os.path.join(base_path, "diffuser_images")
    label_dir = os.path.join(base_path, "ground_truth_lensed")
    if not test:
        csv_path = os.path.join(base_path, "dataset_train.csv")
        dataset = DiffuserMirflickrDataset(csv_path, data_dir, label_dir)
    else:
        csv_path = os.path.join(base_path, "dataset_test.csv")
        dataset = DiffuserMirflickrDataset(csv_path, data_dir, label_dir) 
    return dataset

def create_dataloader(dataset, test=False):
    if not test:
        dataloader = torch.utils.data.DataLoader(
            dataset, num_workers=10, batch_size=40, shuffle=True  # TODO:change batch_size
        )
    else:
        dataloader = torch.utils.data.DataLoader(
            dataset, num_workers=4, batch_size=1, shuffle=False 
        )
    return dataloader

# initialize model for training
if os.path.exists("latest.pt"):
    latest = torch.load("latest.pt")
else:
    latest = None
deconv = initialize_reconstruction(latest=latest)
print(deconv)

# initialize optimizer
opt = initialize_optimizer(deconv, latest=latest)

# initialize training set and validation set
dataset = create_dataset()
    #TODO：change split numbers
dataset, val_dataset = torch.torch.utils.data.random_split(
    dataset, [900, 12], generator=torch.Generator().manual_seed(42)
)
dataloader = create_dataloader(dataset) 
val_dataloader = create_dataloader(val_dataset, test=True)

# initialize iteration count and losses
if latest is not None:
    latest_iter = latest["it"]
    mses = latest["mses"]
    lpips_losses = latest["lpips_losses"]
    losses = latest["losses"]
    validate_mses = latest["validate_mses"]
else:
    latest_iter = 0
    mses = []
    lpips_losses = []
    losses=[]
    validate_mses = []

it = int(latest_iter)

# remove loaded checkpoint
if latest is not None:
    del latest
    torch.cuda.empty_cache()

# create folder for validation data
if not os.path.exists("snapshots/validate/"):
    os.makedirs("snapshots/validate/")
val_dir = "snapshots/validate/"

# run training
train_rgb_recon(
    deconv,
    opt,
    dataloader,
    devices,
    mses,
    num_iterations,
    lpips_weight=lpips_weight,
    lpips_step_milestones=lpips_step_milestones,
    lpips_step_size=lpips_step_size,
    lpips_losses=lpips_losses,
    losses=losses,
    checkpoint_interval = 200,
    snapshot_interval = 300,
    validate_mses=validate_mses,
    validate_args={
        "dataloader": val_dataloader,
        "devices": devices,
        "save_dir": val_dir,
    },
    it=it,
)


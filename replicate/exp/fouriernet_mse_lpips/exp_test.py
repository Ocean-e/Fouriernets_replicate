# fouriernet:test
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
            dataset, num_workers=10, batch_size=40, shuffle=True  
        )
    else:
        dataloader = torch.utils.data.DataLoader(
            dataset, num_workers=4, batch_size=1, shuffle=False 
        )
    return dataloader

def test():
    # initialize model for testing
    if os.path.exists("latest.pt"):
        latest = torch.load("snapshots/state9999.pt")  #TODO:change load state
    else:
        latest = None
    deconv = initialize_reconstruction(latest=latest)
    deconv.eval()
    print(deconv)
    num_params = sum([p.view(-1).shape[0] for p in deconv.parameters()])
    print(num_params)

    # initialize data
    dataset = create_dataset(test=True)
    dataloader = create_dataloader(dataset, test=True)

    # remove loaded checkpoint
    if latest is not None:
        del latest
        torch.cuda.empty_cache()

    # initialize results storage folder
    if not os.path.exists(f'./test'):
        os.mkdir(f'./test')
    save_dir = f'./test'

    losses=test_rgb_recon(deconv, dataloader, devices, save_dir=save_dir)

    return losses

# run test(output:average lpips_loss,mse_loss)
test()
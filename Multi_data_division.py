import os
import sys
import cv2
import yaml
import json
import torch
import argparse
import datetime
import numpy as np
import torch.nn as nn
from torch import optim
from typing import Dict
from objprint import objstr
from sklearn.metrics import accuracy_score
from termcolor import colored
from easydict import EasyDict
import torch.nn.functional as F
from accelerate import Accelerator
from utils.config import create_config
from timm.optim import optim_factory
from utils.common_config import get_train_dataset, get_transformations, \
    get_val_dataset
from utils.MIAutils import give_mia_data, give_data_save, give_all_data

# 加载其中一个模型，以关键字划分数据

def division(dataset,save_name,save_path):
    parser = argparse.ArgumentParser(description='Vanilla Training')
    parser.add_argument('--target_root_dir', default='./files/mia_results/target_results',
                        help='Config file for the environment')
    parser.add_argument('--config_exp', default=f"./files/models/yml/{dataset}/hrnet18/multi_task_baseline.yml",
                        help='Config file for the experiment')
    args = parser.parse_args()

    p = create_config(args.target_root_dir, args.config_exp)

    try:
        ttrain, ttest, strain, stest, tszie, ssize = give_all_data(model_name=save_path + save_name)
        print(colored('Load Data! ', 'blue'))
    except:
        print(colored('New Data! ', 'red'))
        train_transforms, val_transforms = get_transformations(p)
        train_dataset = get_train_dataset(p, train_transforms)
        val_dataset = get_val_dataset(p, val_transforms)
        ttrain, ttest, strain, stest = give_mia_data(train_dataset, val_dataset, batch_size=8, num_workers=0,
                                                     test_row=0.5, shawdow_row=0.5)
        give_data_save(p, ttrain, ttest, strain, stest, save_path=save_path, save_name=save_name)
        ttrain, ttest, strain, stest, tszie, ssize = give_all_data(model_name=save_path + save_name)
    return ttrain, ttest, strain, stest, tszie, ssize
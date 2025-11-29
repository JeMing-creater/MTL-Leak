# This code is referenced from 
# https://github.com/facebookresearch/astmt/
# 
# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
# 
# License: Attribution-NonCommercial 4.0 International

import warnings
import cv2
import os.path
import numpy as np
from PIL import Image
import glob
import math
import torch
import json
import scipy.io as sio
from utils.mypath import MyPath, PROJECT_ROOT_DIR

with open(os.path.join(PROJECT_ROOT_DIR, 'files/db_info/nyu_classes.json')) as f:
    cls_nyu = json.load(f)
with open(os.path.join(PROJECT_ROOT_DIR, 'files/db_info/context_classes.json')) as f:
    cls_context = json.load(f)

def normal_ize(arr):
    arr_norm = np.linalg.norm(arr, ord=2, axis=2)[..., np.newaxis] + 1e-12
    return arr / arr_norm


def eval_normals(loader, folder):
    deg_diff = []
    normals_valid_classes = []
    for cl_nyu in cls_nyu:
        if cl_nyu in cls_context and cl_nyu != 'unknown':
            normals_valid_classes.append(cls_context[cl_nyu])
    used_num = 0
    for i, sample in enumerate(loader):
        # Check for valid labels
        label = sample['normals']
        uniq = np.unique(label)
        if len(uniq) == 1 and uniq[0] == 0:
            continue

        # Load result
        filename = os.path.join(folder, sample['meta']['image'] + '.png')
        if not os.path.exists(filename):
            continue
        pred_test = cv2.imread(filename)
        pred = 2. * cv2.imread(filename).astype(np.float32)[..., ::-1] / 255. - 1
        pred = normal_ize(pred)

        if pred.shape != label.shape:
            warnings.warn('Prediction and ground truth have different size. Resizing Prediction..')
            pred = cv2.resize(pred, label.shape[::-1], interpolation=cv2.INTER_CUBIC)

        valid_mask = (np.linalg.norm(label, ord=2, axis=2) != 0)
        pred[np.invert(valid_mask), :] = 0.
        label[np.invert(valid_mask), :] = 0.
        label = normal_ize(label)

        deg_diff_tmp = np.rad2deg(np.arccos(np.clip(np.sum(pred * label, axis=2), a_min=-1, a_max=1)))
        deg_diff.extend(deg_diff_tmp[valid_mask])
        used_num += 1
    deg_diff = np.array(deg_diff)
    eval_result = dict()
    eval_result['mean'] = np.mean(deg_diff)
    eval_result['median'] = np.median(deg_diff)
    eval_result['rmse'] = np.mean(deg_diff ** 2) ** 0.5
    eval_result['11.25'] = np.mean(deg_diff < 11.25) * 100
    eval_result['22.5'] = np.mean(deg_diff < 22.5) * 100
    eval_result['30'] = np.mean(deg_diff < 30) * 100

    eval_result = {x: eval_result[x].tolist() for x in eval_result}

    return eval_result


class NormalsMeter(object):
    def __init__(self):
        self.eval_dict = {'mean': 0., 'rmse': 0., '11.25': 0., '22.5': 0., '30': 0., 'n': 0}

    @torch.no_grad()
    def update(self, pred, gt):
        # Performance measurement happens in pixel wise fashion (Same as code from ASTMT (above))
        pred = 2 * pred / 255 - 1 
        pred = pred.permute(0, 3, 1, 2) # [B, C, H, W]
        valid_mask = (gt != 255)
        invalid_mask = (gt == 255)

        # Put zeros where mask is invalid
        pred[invalid_mask] = 0.0
        gt[invalid_mask] = 0.0
        
        # Calculate difference expressed in degrees 
        deg_diff_tmp = (180 / math.pi) * (torch.acos(torch.clamp(torch.sum(pred * gt, 1), min=-1, max=1)))
        deg_diff_tmp = torch.masked_select(deg_diff_tmp, valid_mask[:,0])

        self.eval_dict['mean'] += torch.sum(deg_diff_tmp).item()
        self.eval_dict['rmse'] += torch.sum(torch.sqrt(torch.pow(deg_diff_tmp, 2))).item()
        self.eval_dict['11.25'] += torch.sum((deg_diff_tmp < 11.25).float()).item() * 100
        self.eval_dict['22.5'] += torch.sum((deg_diff_tmp < 22.5).float()).item() * 100
        self.eval_dict['30'] += torch.sum((deg_diff_tmp < 30).float()).item() * 100
        self.eval_dict['n'] += deg_diff_tmp.numel()

    def reset(self):
        self.eval_dict = {'mean': 0., 'rmse': 0., '11.25': 0., '22.5': 0., '30': 0., 'n': 0}

    def get_score(self, verbose=True):
        eval_result = dict()
        eval_result['mean'] = self.eval_dict['mean'] / self.eval_dict['n']
        eval_result['rmse'] = self.eval_dict['mean'] / self.eval_dict['n']
        eval_result['11.25'] = self.eval_dict['11.25'] / self.eval_dict['n']
        eval_result['22.5'] = self.eval_dict['22.5'] / self.eval_dict['n']
        eval_result['30'] = self.eval_dict['30'] / self.eval_dict['n']

        if verbose:
            print('Results for Surface Normal Estimation')
            for x in eval_result:
                spaces = ""
                for j in range(0, 15 - len(x)):
                    spaces += ' '
                print('{0:s}{1:s}{2:.4f}'.format(x, spaces, eval_result[x]))

        return eval_result


def eval_normals_predictions(database, save_dir, overfit=False,definition_db=None):
    """ Evaluate the normals maps that are stored in the save dir """
    # Dataloaders
    if database == 'PASCALContext':
        from utils.pascal_context import PASCALContext
        gt_set = 'val'
        db = PASCALContext(split=gt_set, do_edge=False, do_human_parts=False, do_semseg=False,
                                          do_normals=True, overfit=overfit)
        if definition_db !=None:
            gt_set = ['train']
            tdb = PASCALContext(split=gt_set, do_edge=False, do_human_parts=False, do_semseg=False,
                               do_normals=True, overfit=overfit)
            db = db+tdb
    elif database == 'NYUD':
        from utils.common_config import NYUD_MT
        gt_set = 'val'
        db = NYUD_MT(split=gt_set, do_normals=True, overfit=overfit)
        if definition_db != None:
            gt_set = 'train'
            tdb = NYUD_MT(split=gt_set, do_normals=True, overfit=overfit)
            db = db + tdb
    else:
        raise NotImplementedError

    base_name = database + '_' + 'test' + '_normals'
    fname = os.path.join(save_dir, base_name + '.json')

    # Eval the model
    print('Evaluate the saved images (surface normals)') 
    eval_results = eval_normals(db, os.path.join(save_dir, 'normals'))
    with open(fname, 'w') as f:
        json.dump(eval_results, f)

    # Print results
    print('Results for Surface Normal Estimation')
    for x in eval_results:
        spaces = ""
        for j in range(0, 15 - len(x)):
            spaces += ' '
        print('{0:s}{1:s}{2:.4f}'.format(x, spaces, eval_results[x]))

    return eval_results

#
# Authors: Simon Vandenhende
# Licensed under the CC BY-NC 4.0 license (https://creativecommons.org/licenses/by-nc/4.0/)

import os
import cv2
import imageio
import numpy as np
import json
import torch
import scipy.io as sio
import math
import warnings
import cv2
import os.path
import glob
import json
import torch
from PIL import Image

import torch.nn.functional as F

VOC_CATEGORY_NAMES = ['background',
                      'aeroplane', 'bicycle', 'bird', 'boat', 'bottle',
                      'bus', 'car', 'cat', 'chair', 'cow',
                      'diningtable', 'dog', 'horse', 'motorbike', 'person',
                      'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor']


NYU_CATEGORY_NAMES = ['wall', 'floor', 'cabinet', 'bed', 'chair',
                      'sofa', 'table', 'door', 'window', 'bookshelf',
                      'picture', 'counter', 'blinds', 'desk', 'shelves',
                      'curtain', 'dresser', 'pillow', 'mirror', 'floor mat',
                      'clothes', 'ceiling', 'books', 'refridgerator', 'television',
                      'paper', 'towel', 'shower curtain', 'box', 'whiteboard',
                      'person', 'night stand', 'toilet', 'sink', 'lamp',
                      'bathtub', 'bag', 'otherstructure', 'otherfurniture', 'otherprop']


PART_CATEGORY_NAMES = ['background', 'head', 'torso', 'uarm', 'larm', 'uleg', 'lleg']

class SemsegMeter(object):
    def __init__(self, database):
        if database == 'PASCALContext':
            n_classes = 20
            cat_names = VOC_CATEGORY_NAMES
            has_bg = True

        elif database == 'NYUD':
            n_classes = 40
            cat_names = NYU_CATEGORY_NAMES
            has_bg = False

        else:
            raise NotImplementedError

        self.n_classes = n_classes + int(has_bg)
        self.cat_names = cat_names
        self.tp = [0] * self.n_classes
        self.fp = [0] * self.n_classes
        self.fn = [0] * self.n_classes

    @torch.no_grad()
    def update(self, pred, gt):
        pred = pred.squeeze()
        gt = gt.squeeze()
        valid = (gt != 255)

        for i_part in range(0, self.n_classes):
            tmp_gt = (gt == i_part)
            tmp_pred = (pred == i_part)
            self.tp[i_part] += torch.sum(tmp_gt & tmp_pred & valid).item()
            self.fp[i_part] += torch.sum(~tmp_gt & tmp_pred & valid).item()
            self.fn[i_part] += torch.sum(tmp_gt & ~tmp_pred & valid).item()

    def reset(self):
        self.tp = [0] * self.n_classes
        self.fp = [0] * self.n_classes
        self.fn = [0] * self.n_classes

    def get_score(self, verbose=True):
        jac = [0] * self.n_classes
        for i_part in range(self.n_classes):
            jac[i_part] = float(self.tp[i_part]) / max(float(self.tp[i_part] + self.fp[i_part] + self.fn[i_part]), 1e-8)

        eval_result = dict()
        eval_result['jaccards_all_categs'] = jac
        eval_result['mIoU'] = np.mean(jac)

        if verbose:
            print('\nSemantic Segmentation mIoU: {0:.4f}\n'.format(100 * eval_result['mIoU']))
            class_IoU = eval_result['jaccards_all_categs']
            for i in range(len(class_IoU)):
                spaces = ''
                for j in range(0, 20 - len(self.cat_names[i])):
                    spaces += ' '
                print('{0:s}{1:s}{2:.4f}'.format(self.cat_names[i], spaces, 100 * class_IoU[i]))

        return eval_result


class DepthMeter(object):
    def __init__(self):
        self.total_rmses = 0.0
        self.total_log_rmses = 0.0
        self.n_valid = 0.0

    @torch.no_grad()
    def update(self, pred, gt):
        pred, gt = pred.squeeze(), gt.squeeze()

        # Determine valid mask
        mask = (gt != 255).bool()
        self.n_valid += mask.float().sum().item()  # Valid pixels per image

        # Only positive depth values are possible
        pred = torch.clamp(pred, min=1e-9)

        # Per pixel rmse and log-rmse.
        log_rmse_tmp = torch.pow(torch.log(gt) - torch.log(pred), 2)
        log_rmse_tmp = torch.masked_select(log_rmse_tmp, mask)
        self.total_log_rmses += log_rmse_tmp.sum().item()

        rmse_tmp = torch.pow(gt - pred, 2)
        rmse_tmp = torch.masked_select(rmse_tmp, mask)
        self.total_rmses += rmse_tmp.sum().item()

    def reset(self):
        self.rmses = []
        self.log_rmses = []

    def get_score(self, verbose=True):
        eval_result = dict()
        eval_result['rmse'] = np.sqrt(self.total_rmses / self.n_valid)
        eval_result['log_rmse'] = np.sqrt(self.total_log_rmses / self.n_valid)

        if verbose:
            print('Results for depth prediction')
            for x in eval_result:
                spaces = ''
                for j in range(0, 15 - len(x)):
                    spaces += ' '
                print('{0:s}{1:s}{2:.4f}'.format(x, spaces, eval_result[x]))

        return eval_result


class SaliencyMeter(object):
    def __init__(self):
        self.mask_thres = np.linspace(0.2, 0.9, 15)  # As below
        self.all_jacards = []
        self.prec = []
        self.rec = []

    @torch.no_grad()
    def update(self, pred, gt):
        # Predictions and ground-truth
        b = pred.size(0)
        pred = pred.float().squeeze() / 255.
        gt = gt.squeeze().cpu().numpy()

        # Allocate memory for batch results
        jaccards = np.zeros((b, len(self.mask_thres)))
        prec = np.zeros((b, len(self.mask_thres)))
        rec = np.zeros((b, len(self.mask_thres)))

        for j, thres in enumerate(self.mask_thres):
            # gt_eval = (gt > thres).cpu().numpy() # Removed this from ASTMT code. GT is already binarized.
            mask_eval = (pred > thres).cpu().numpy()
            for i in range(b):
                jaccards[i, j] = jaccard(gt[i], mask_eval[i])
                prec[i, j], rec[i, j] = precision_recall(gt[i], mask_eval[i])

        self.all_jacards.append(jaccards)
        self.prec.append(prec)
        self.rec.append(rec)

    def reset(self):
        self.all_jacards = []
        self.prec = []
        self.rec = []

    def get_score(self, verbose=True):
        eval_result = dict()

        # Concatenate batched results
        eval_result['all_jaccards'] = np.concatenate(self.all_jacards)
        eval_result['prec'] = np.concatenate(self.prec)
        eval_result['rec'] = np.concatenate(self.rec)

        # Average for each threshold
        eval_result['mIoUs'] = np.mean(eval_result['all_jaccards'], 0)

        eval_result['mPrec'] = np.mean(eval_result['prec'], 0)
        eval_result['mRec'] = np.mean(eval_result['rec'], 0)
        eval_result['F'] = 2 * eval_result['mPrec'] * eval_result['mRec'] / \
                           (eval_result['mPrec'] + eval_result['mRec'] + 1e-12)

        # Maximum of averages (maxF, maxmIoU)
        eval_result['mIoU'] = np.max(eval_result['mIoUs'])
        eval_result['maxF'] = np.max(eval_result['F'])

        eval_result = {x: eval_result[x].tolist() for x in eval_result}

        # if verbose:
        #     # Print the results
        #     print('Results for Saliency Estimation')
        #     print('mIoU: {0:.3f}'.format(100 * eval_result['mIoU']))
        #     print('maxF: {0:.3f}'.format(100 * eval_result['maxF']))

        return eval_result


class HumanPartsMeter(object):
    def __init__(self, database):
        assert (database == 'PASCALContext')
        self.database = database
        self.cat_names = PART_CATEGORY_NAMES
        self.n_parts = 6
        self.tp = [0] * (self.n_parts + 1)
        self.fp = [0] * (self.n_parts + 1)
        self.fn = [0] * (self.n_parts + 1)

    @torch.no_grad()
    def update(self, pred, gt):
        pred, gt = pred.squeeze(), gt.squeeze()
        valid = (gt != 255)

        for i_part in range(self.n_parts + 1):
            tmp_gt = (gt == i_part)
            tmp_pred = (pred == i_part)
            self.tp[i_part] += torch.sum(tmp_gt & tmp_pred & (valid)).item()
            self.fp[i_part] += torch.sum(~tmp_gt & tmp_pred & (valid)).item()
            self.fn[i_part] += torch.sum(tmp_gt & ~tmp_pred & (valid)).item()

    def reset(self):
        self.tp = [0] * (self.n_parts + 1)
        self.fp = [0] * (self.n_parts + 1)
        self.fn = [0] * (self.n_parts + 1)

    def get_score(self, verbose=True):
        jac = [0] * (self.n_parts + 1)
        for i_part in range(0, self.n_parts + 1):
            jac[i_part] = float(self.tp[i_part]) / max(float(self.tp[i_part] + self.fp[i_part] + self.fn[i_part]), 1e-8)

        eval_result = dict()
        eval_result['jaccards_all_categs'] = jac
        eval_result['mIoU'] = np.mean(jac)

        # print('\nHuman Parts mIoU: {0:.4f}\n'.format(100 * eval_result['mIoU']))
        class_IoU = jac
        for i in range(len(class_IoU)):
            spaces = ''
            for j in range(0, 15 - len(self.cat_names[i])):
                spaces += ' '
            # print('{0:s}{1:s}{2:.4f}'.format(self.cat_names[i], spaces, 100 * class_IoU[i]))

        return eval_result


class NormalsMeter(object):
    def __init__(self):
        self.eval_dict = {'mean': 0., 'rmse': 0., '11.25': 0., '22.5': 0., '30': 0., 'n': 0}

    @torch.no_grad()
    def update(self, pred, gt):
        # Performance measurement happens in pixel wise fashion (Same as code from ASTMT (above))
        pred = 2 * pred / 255 - 1
        pred = pred.permute(0, 3, 1, 2)  # [B, C, H, W]
        valid_mask = (gt != 255)
        invalid_mask = (gt == 255)

        # Put zeros where mask is invalid
        pred[invalid_mask] = 0.0
        gt[invalid_mask] = 0.0

        # Calculate difference expressed in degrees
        deg_diff_tmp = (180 / math.pi) * (torch.acos(torch.clamp(torch.sum(pred * gt, 1), min=-1, max=1)))
        deg_diff_tmp = torch.masked_select(deg_diff_tmp, valid_mask[:, 0])

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
        if self.eval_dict['n'] == 0:
            self.eval_dict['n'] = 1
        eval_result['mean'] = self.eval_dict['mean'] / self.eval_dict['n']
        eval_result['rmse'] = self.eval_dict['mean'] / self.eval_dict['n']
        eval_result['11.25'] = self.eval_dict['11.25'] / self.eval_dict['n']
        eval_result['22.5'] = self.eval_dict['22.5'] / self.eval_dict['n']
        eval_result['30'] = self.eval_dict['30'] / self.eval_dict['n']

        # if verbose:
        #     print('Results for Surface Normal Estimation')
        #     for x in eval_result:
        #         spaces = ""
        #         for j in range(0, 15 - len(x)):
        #             spaces += ' '
        #         print('{0:s}{1:s}{2:.4f}'.format(x, spaces, eval_result[x]))

        return eval_result


def jaccard(gt, pred, void_pixels=None):
    assert (gt.shape == pred.shape)

    if void_pixels is None:
        void_pixels = np.zeros_like(gt)
    assert (void_pixels.shape == gt.shape)

    gt = gt.astype(np.bool_)
    pred = pred.astype(np.bool_)
    void_pixels = void_pixels.astype(np.bool_)
    if np.isclose(np.sum(gt & np.logical_not(void_pixels)), 0) and np.isclose(
            np.sum(pred & np.logical_not(void_pixels)), 0):
        return 1

    else:
        return np.sum(((gt & pred) & np.logical_not(void_pixels))) / \
               np.sum(((gt | pred) & np.logical_not(void_pixels)), dtype=np.float32)


def precision_recall(gt, pred, void_pixels=None):
    if void_pixels is None:
        void_pixels = np.zeros_like(gt)

    gt = gt.astype(np.bool_)
    pred = pred.astype(np.bool_)
    void_pixels = void_pixels.astype(np.bool_)

    tp = ((pred & gt) & ~void_pixels).sum()
    fn = ((~pred & gt) & ~void_pixels).sum()

    fp = ((pred & ~gt) & ~void_pixels).sum()

    prec = tp / (tp + fp + 1e-12)
    rec = tp / (tp + fn + 1e-12)

    return prec, rec


def calculate_multi_task_performance(eval_dict):
    tasks = eval_dict.keys()
    num_tasks = len(tasks)
    mtl_performance = 0.0

    for task in tasks:
        mtl = eval_dict[task]

        if task == 'depth':  # rmse lower is better
            mtl_performance -= mtl['rmse']

        elif task in ['semseg', 'sal', 'human_parts']:  # mIoU higher is better
            mtl_performance += mtl['mIoU']

        elif task == 'normals':  # mean error lower is better
            mtl_performance -= mtl['mean']

        elif task == 'edge':  # odsF higher is better
            mtl_performance += mtl['odsF']
        else:
            raise NotImplementedError

    return mtl_performance


def eval_all_results(p, model, test, accelerator):
    if p['dataset'] == 'nyud':
        database = 'NYUD'
    else:
        database = 'PASCALContext'
    segmeter = SemsegMeter(database)
    hmmeter = HumanPartsMeter(database='PASCALContext')
    depmeter = DepthMeter()
    salmeter = SaliencyMeter()
    nometer = NormalsMeter()

    # save_dir = p['save_dir']
    results = {}
    se_results = {}
    for batch in test:
        images = batch['image'].to(accelerator.device)
        targets = {task: batch[task].to(accelerator.device) for task in p.ALL_TASKS.NAMES}
        output = model(images)
        if 'semseg' in p.TASKS.NAMES:
            segmeter.update(get_output(output['semseg'], 'semseg'), targets['semseg'])
        if 'depth' in p.TASKS.NAMES:
            depmeter.update(get_output(output['depth'], 'depth'), targets['depth'])
        if 'sal' in p.TASKS.NAMES:
            salmeter.update(get_output(output['sal'], 'sal'), targets['sal'])
        if 'human_parts' in p.TASKS.NAMES:
            hmmeter.update(get_output(output['human_parts'], 'human_parts'), targets['human_parts'])
        if 'normals' in p.TASKS.NAMES:
            nometer.update(get_output(output['normals'], 'normals'), targets['normals'])

    if 'semseg' in p.TASKS.NAMES:
        results['semseg'] = segmeter.get_score(verbose=False)
        miou = results['semseg']['mIoU']
        accelerator.print(f"semseg MIOU: {miou}", flush=True)
    if 'depth' in p.TASKS.NAMES:
        results['depth'] = depmeter.get_score(verbose=False)
        rmse = results['depth']['rmse']
        accelerator.print(f"depth Rmse: {rmse}", flush=True)
    if 'human_parts' in p.TASKS.NAMES:
        results['human_parts'] = hmmeter.get_score(verbose=False)
        miou = results['human_parts']['mIoU']
        accelerator.print(f"human_parts MIOU: {miou}", flush=True)
    if 'sal' in p.TASKS.NAMES:
        results['sal'] = salmeter.get_score(verbose=False)
        miou = results['sal']['mIoU']
        accelerator.print(f"sal MIOU: {miou}", flush=True)
    if 'normals' in p.TASKS.NAMES:
        results['normals'] = nometer.get_score(verbose=False)
        miou = results['normals']['mean']
        accelerator.print(f"normals mErr: {miou}", flush=True)

    if p['setup'] == 'multi_task':
        results['multi_task_performance'] = calculate_multi_task_performance(results)
        print('Multi-task learning performance on test set is %.2f' % (results['multi_task_performance']))

    return results


def validate_results(p, current, reference):
    """
        Compare the results between the current eval dict and a reference eval dict.
        Returns a tuple (boolean, eval_dict).
        The boolean is true if the current eval dict has higher performance compared
        to the reference eval dict.
        The returned eval dict is the one with the highest performance.
    """
    tasks = p.TASKS.NAMES
    
    if len(tasks) == 1: # Single-task performance
        task = tasks[0]
        if task == 'semseg': # Semantic segmentation (mIoU)
            if current['semseg']['mIoU'] > reference['semseg']['mIoU']:
                print('New best semgentation model %.2f -> %.2f' %(100*reference['semseg']['mIoU'], 100*current['semseg']['mIoU']))
                improvement = True
            else:
                print('No new best semgentation model %.2f -> %.2f' %(100*reference['semseg']['mIoU'], 100*current['semseg']['mIoU']))
                improvement = False
        
        elif task == 'human_parts': # Human parts segmentation (mIoU)
            if current['human_parts']['mIoU'] > reference['human_parts']['mIoU']:
                print('New best human parts semgentation model %.2f -> %.2f' %(100*reference['human_parts']['mIoU'], 100*current['human_parts']['mIoU']))
                improvement = True
            else:
                print('No new best human parts semgentation model %.2f -> %.2f' %(100*reference['human_parts']['mIoU'], 100*current['human_parts']['mIoU']))
                improvement = False

        elif task == 'sal': # Saliency estimation (mIoU)
            if current['sal']['mIoU'] > reference['sal']['mIoU']:
                print('New best saliency estimation model %.2f -> %.2f' %(100*reference['sal']['mIoU'], 100*current['sal']['mIoU']))
                improvement = True
            else:
                print('No new best saliency estimation model %.2f -> %.2f' %(100*reference['sal']['mIoU'], 100*current['sal']['mIoU']))
                improvement = False

        elif task == 'depth': # Depth estimation (rmse)
            if current['depth']['rmse'] < reference['depth']['rmse']:
                print('New best depth estimation model %.3f -> %.3f' %(reference['depth']['rmse'], current['depth']['rmse']))
                improvement = True
            else:
                print('No new best depth estimation model %.3f -> %.3f' %(reference['depth']['rmse'], current['depth']['rmse']))
                improvement = False
        
        elif task == 'normals': # Surface normals (mean error)
            if current['normals']['mean'] < reference['normals']['mean']:
                print('New best surface normals estimation model %.3f -> %.3f' %(reference['normals']['mean'], current['normals']['mean']))
                improvement = True
            else:
                print('No new best surface normals estimation model %.3f -> %.3f' %(reference['normals']['mean'], current['normals']['mean']))
                improvement = False

        elif task == 'edge': # Validation happens based on odsF
            if current['edge']['odsF'] > reference['edge']['odsF']:
                print('New best edge detection model %.3f -> %.3f' %(reference['edge']['odsF'], current['edge']['odsF']))
                improvement = True
            
            else:
                print('No new best edge detection model %.3f -> %.3f' %(reference['edge']['odsF'], current['edge']['odsF']))
                improvement = False


    else: # Multi-task performance
        if current['multi_task_performance'] > reference['multi_task_performance']:
            print('New best multi-task model %.2f -> %.2f' %(reference['multi_task_performance'], current['multi_task_performance']))
            improvement = True

        else:
            print('No new best multi-task model %.2f -> %.2f' %(reference['multi_task_performance'], current['multi_task_performance']))
            improvement = False

    if improvement: # Return result
        return True, current

    else:
        return False, reference


def get_output(output, task):
    output = output.permute(0, 2, 3, 1)

    if task == 'normals':
        output = (F.normalize(output, p=2, dim=3) + 1.0) * 255 / 2.0

    elif task in {'semseg', 'human_parts'}:
        _, output = torch.max(output, dim=3)

    elif task in {'edge', 'sal'}:
        output = torch.squeeze(255 * 1 / (1 + torch.exp(-output)))

    elif task in {'depth'}:
        pass

    else:
        raise ValueError('Select one of the valid tasks')

    return output




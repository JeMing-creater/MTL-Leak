#
# Authors: Simon Vandenhende
# Licensed under the CC BY-NC 4.0 license (https://creativecommons.org/licenses/by-nc/4.0/)

import os
import copy
import torch
import torch.nn.functional as F
import numpy as np
import requests
from torchvision import transforms
from torch.utils.data import DataLoader
from utils.custom_collate import collate_mil
import os
import sys
import tarfile
import cv2
from PIL import Image
import torch.utils.data as data
import scipy.io as sio
from six.moves import urllib
from utils.mypath import MyPath
from utils.utils import mkdir_if_missing
import tarfile
import json
from skimage.morphology import thin
from six.moves import urllib

"""
    Model getters 
"""

PROJECT_ROOT_DIR = os.path.dirname(os.path.abspath(__file__)).split('/')[0]


def get_backbone(p):
    """ Return the backbone """
    if p['backbone'] == 'resnet18':
        from files.models.resnet import resnet18
        backbone = resnet18(p['backbone_kwargs']['pretrained'])
        backbone_channels = 512

    elif p['backbone'] == 'resnet50':
        from files.models.resnet import resnet50
        backbone = resnet50(p['backbone_kwargs']['pretrained'])
        backbone_channels = 2048

    elif p['backbone'] == 'hrnet_w18':
        from files.models.seg_hrnet import hrnet_w18
        backbone = hrnet_w18(p['dataset'], p['backbone_kwargs']['pretrained'])
        backbone_channels = [18, 36, 72, 144]

    else:
        raise NotImplementedError

    if p['backbone_kwargs']['dilated']:  # Add dilated convolutions
        assert (p['backbone'] in ['hrnet_w18', 'resnet18', 'resnet50'])
        from files.models.resnet_dilated import ResnetDilated
        backbone = ResnetDilated(backbone)

    if 'fuse_hrnet' in p['backbone_kwargs'] and p['backbone_kwargs'][
        'fuse_hrnet']:  # Fuse the multi-scale HRNet features
        from files.models.seg_hrnet import HighResolutionFuse
        backbone = torch.nn.Sequential(backbone, HighResolutionFuse(backbone_channels, 256))
        backbone_channels = sum(backbone_channels)

    return backbone, backbone_channels


def get_head(p, backbone_channels, task, model_head=None):
    """ Return the decoder head """
    if model_head != None:
        model_head = model_head
    else:
        model_head = p['head']
    if model_head == 'deeplab':
        from files.models.aspp import DeepLabHead
        return DeepLabHead(backbone_channels, p.TASKS.NUM_OUTPUT[task])

    elif model_head == 'hrnet':
        from files.models.seg_hrnet import HighResolutionHead
        return HighResolutionHead(backbone_channels, p.TASKS.NUM_OUTPUT[task])

    else:
        raise NotImplementedError


def get_model(p):
    """ Return the model """
    backbone, backbone_channels = get_backbone(p)

    if p['setup'] == 'single_task':
        from files.models.models import SingleTaskModel
        task = p.TASKS.NAMES[0]
        head = get_head(p, backbone_channels, task)
        model = SingleTaskModel(backbone, head, task)

    elif p['setup'] == 'multi_task':
        if p['model'] == 'baseline':
            from files.models.models import MultiTaskModel
            heads = torch.nn.ModuleDict({task: get_head(p, backbone_channels, task) for task in p.TASKS.NAMES})
            model = MultiTaskModel(backbone, heads, p.TASKS.NAMES)


        elif p['model'] == 'cross_stitch':
            from files.models.models import SingleTaskModel
            from files.models.cross_stitch import CrossStitchNetwork

            # Load single-task models
            backbone_dict, decoder_dict = {}, {}
            for task in p.TASKS.NAMES:
                model = SingleTaskModel(copy.deepcopy(backbone), get_head(p, backbone_channels, task, ), task)
                model = torch.nn.DataParallel(model)
                checkpoint = torch.load(
                    os.path.join(p['root_dir'], p['train_db_name'], p['backbone'], 'single_task', task,
                                 'best_model.pth.tar'))
                if p['debug'] == True:
                    new_checkpoint = {}
                    for key in checkpoint.keys():
                        new_checkpoint['module.' + key] = checkpoint[key]
                    checkpoint = new_checkpoint
                model.load_state_dict(checkpoint)
                backbone_dict[task] = model.module.backbone
                decoder_dict[task] = model.module.decoder

            # Stitch the single-task models together
            model = CrossStitchNetwork(p, torch.nn.ModuleDict(backbone_dict), torch.nn.ModuleDict(decoder_dict),
                                       **p['model_kwargs']['cross_stitch_kwargs'])


        elif p['model'] == 'nddr_cnn':
            from files.models.models import SingleTaskModel
            from files.models.nddr_cnn import NDDRCNN
            # backbone_channels = [18, 36, 72, 144]
            # Load single-task models
            backbone_dict, decoder_dict = {}, {}
            for task in p.TASKS.NAMES:
                model = SingleTaskModel(copy.deepcopy(backbone), get_head(p, backbone_channels, task), task)
                model = torch.nn.DataParallel(model)
                checkpoint = torch.load(
                    os.path.join(p['root_dir'], p['train_db_name'], p['backbone'], 'single_task', task,
                                 'best_model.pth.tar'))
                if p['debug'] == True:
                    new_checkpoint = {}
                    for key in checkpoint.keys():
                        new_checkpoint['module.' + key] = checkpoint[key]
                    checkpoint = new_checkpoint
                model.load_state_dict(checkpoint)
                backbone_dict[task] = model.module.backbone
                decoder_dict[task] = model.module.decoder

            # Stitch the single-task models together
            model = NDDRCNN(p, torch.nn.ModuleDict(backbone_dict), torch.nn.ModuleDict(decoder_dict),
                            **p['model_kwargs']['nddr_cnn_kwargs'])


        elif p['model'] == 'mtan':
            from files.models.mtan import MTAN
            heads = torch.nn.ModuleDict({task: get_head(p, backbone_channels, task) for task in p.TASKS.NAMES})
            model = MTAN(p, backbone, heads, **p['model_kwargs']['mtan_kwargs'])


        elif p['model'] == 'pad_net':
            from files.models.padnet import PADNet
            model = PADNet(p, backbone, backbone_channels)


        elif p['model'] == 'mti_net':
            from files.models.mti_net import MTINet
            heads = torch.nn.ModuleDict({task: get_head(p, backbone_channels, task) for task in p.TASKS.NAMES})
            model = MTINet(p, backbone, backbone_channels, heads)


        else:
            raise NotImplementedError('Unknown model {}'.format(p['model']))


    else:
        raise NotImplementedError('Unknown setup {}'.format(p['setup']))

    return model


"""
    Transformations, datasets and dataloaders
"""


def get_transformations(p):
    """ Return transformations for training and evaluationg """
    import utils.custom_transforms as tr

    # Training transformations
    if p['train_db_name'] == 'NYUD':
        # Horizontal flips with probability of 0.5
        transforms_tr = [tr.RandomHorizontalFlip()]

        # Rotations and scaling
        transforms_tr.extend([tr.ScaleNRotate(rots=[0], scales=[1.0, 1.2, 1.5],
                                              flagvals={x: p.ALL_TASKS.FLAGVALS[x] for x in p.ALL_TASKS.FLAGVALS})])

    elif p['train_db_name'] == 'PASCALContext':
        # Horizontal flips with probability of 0.5
        transforms_tr = [tr.RandomHorizontalFlip()]

        # Rotations and scaling
        transforms_tr.extend([tr.ScaleNRotate(rots=(-20, 20), scales=(.75, 1.25),
                                              flagvals={x: p.ALL_TASKS.FLAGVALS[x] for x in p.ALL_TASKS.FLAGVALS})])

    else:
        raise ValueError('Invalid train db name'.format(p['train_db_name']))

    # Fixed Resize to input resolution
    transforms_tr.extend([tr.FixedResize(resolutions={x: tuple(p.TRAIN.SCALE) for x in p.ALL_TASKS.FLAGVALS},
                                         flagvals={x: p.ALL_TASKS.FLAGVALS[x] for x in p.ALL_TASKS.FLAGVALS})])
    transforms_tr.extend([tr.AddIgnoreRegions(), tr.ToTensor(),
                          tr.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
    transforms_tr = transforms.Compose(transforms_tr)

    # Testing (during training transforms)
    transforms_ts = []
    transforms_ts.extend([tr.FixedResize(resolutions={x: tuple(p.TEST.SCALE) for x in p.TASKS.FLAGVALS},
                                         flagvals={x: p.TASKS.FLAGVALS[x] for x in p.TASKS.FLAGVALS})])
    transforms_ts.extend([tr.AddIgnoreRegions(), tr.ToTensor(),
                          tr.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
    transforms_ts = transforms.Compose(transforms_ts)

    return transforms_tr, transforms_ts


def download_file_from_google_drive(id, destination):
    URL = "https://drive.google.com/u/1/uc?export=download"
    CHUNK_SIZE = 32768
    session = requests.Session()

    response = session.get(URL, params={'id': id}, stream=True)

    for key, value in response.cookies.items():
        if key.startswith('download_warning'):
            token = value

    if token:
        params = {'id': id, 'confirm': token}
        response = session.get(URL, params=params, stream=True)

    with open(destination, "wb") as f:
        for chunk in response.iter_content(CHUNK_SIZE):
            if chunk:  # filter out keep-alive new chunks
                f.write(chunk)


class NYUD_MT(data.Dataset):
    GOOGLE_DRIVE_ID = '14EAEMXmd3zs2hIMY63UhHPSFPDAkiTzw'
    FILE = 'NYUD_MT.tgz'

    def __init__(self,
                 root=MyPath.db_root_dir('NYUD_MT'),
                 download=True,
                 split='val',
                 transform=None,
                 retname=True,
                 overfit=False,
                 do_edge=False,
                 do_semseg=False,
                 do_normals=False,
                 do_depth=False,
                 ):

        self.root = root

        if download:
            self._download()

        self.transform = transform

        if isinstance(split, str):
            self.split = [split]
        else:
            split.sort()
            self.split = split

        self.retname = retname

        # Original Images
        self.im_ids = []
        self.images = []
        _image_dir = os.path.join(root, 'images')

        # Edge Detection
        self.do_edge = do_edge
        self.edges = []
        _edge_gt_dir = os.path.join(root, 'edge')

        # Semantic segmentation
        self.do_semseg = do_semseg
        self.semsegs = []
        _semseg_gt_dir = os.path.join(root, 'segmentation')

        # Surface Normals
        self.do_normals = do_normals
        self.normals = []
        _normal_gt_dir = os.path.join(root, 'normals')

        # Depth
        self.do_depth = do_depth
        self.depths = []
        _depth_gt_dir = os.path.join(root, 'depth')

        # train/val/test splits are pre-cut
        _splits_dir = os.path.join(root, 'gt_sets')

        print('Initializing dataloader for NYUD {} set'.format(''.join(self.split)))
        for splt in self.split:
            with open(os.path.join(os.path.join(_splits_dir, splt + '.txt')), 'r') as f:
                lines = f.read().splitlines()

            for ii, line in enumerate(lines):
                # Images
                _image = os.path.join(_image_dir, line + '.jpg')
                assert os.path.isfile(_image)
                self.images.append(_image)
                self.im_ids.append(line.rstrip('\n'))

                # Edges
                _edge = os.path.join(_edge_gt_dir, line + '.npy')
                assert os.path.isfile(_edge)
                self.edges.append(_edge)

                # Semantic Segmentation
                _semseg = os.path.join(_semseg_gt_dir, line + '.png')
                assert os.path.isfile(_semseg)
                self.semsegs.append(_semseg)

                # Surface Normals
                _normal = os.path.join(_normal_gt_dir, line + '.npy')
                assert os.path.isfile(_normal)
                self.normals.append(_normal)

                # Depth Prediction
                _depth = os.path.join(_depth_gt_dir, line + '.npy')
                assert os.path.isfile(_depth)
                self.depths.append(_depth)

        if self.do_edge:
            assert (len(self.images) == len(self.edges))
        if self.do_semseg:
            assert (len(self.images) == len(self.semsegs))
        if self.do_depth:
            assert (len(self.images) == len(self.depths))
        if self.do_normals:
            assert (len(self.images) == len(self.normals))

        # Uncomment to overfit to one image
        if overfit:
            n_of = 64
            self.images = self.images[:n_of]
            self.im_ids = self.im_ids[:n_of]

    def __getitem__(self, index):
        sample = {}

        _img = self._load_img(index)
        sample['image'] = _img

        if self.do_edge:
            _edge = self._load_edge(index)
            if _edge.shape != _img.shape[:2]:
                _edge = cv2.resize(_edge, _img.shape[:2][::-1], interpolation=cv2.INTER_NEAREST)
            sample['edge'] = _edge

        if self.do_semseg:
            _semseg = self._load_semseg(index)
            if _semseg.shape != _img.shape[:2]:
                print('RESHAPE SEMSEG')
                _semseg = cv2.resize(_semseg, _img.shape[:2][::-1], interpolation=cv2.INTER_NEAREST)
            sample['semseg'] = _semseg

        if self.do_normals:
            _normals = self._load_normals(index)
            if _normals.shape[:2] != _img.shape[:2]:
                _normals = cv2.resize(_normals, _img.shape[:2][::-1], interpolation=cv2.INTER_CUBIC)
            sample['normals'] = _normals

        if self.do_depth:
            _depth = self._load_depth(index)
            if _depth.shape[:2] != _img.shape[:2]:
                print('RESHAPE DEPTH')
                _depth = cv2.resize(_depth, _img.shape[:2][::-1], interpolation=cv2.INTER_NEAREST)
            sample['depth'] = _depth

        if self.retname:
            sample['meta'] = {'image': str(self.im_ids[index]),
                              'im_size': (_img.shape[0], _img.shape[1])}

        if self.transform is not None:
            sample = self.transform(sample)

        return sample

    def __len__(self):
        return len(self.images)

    def _load_img(self, index):
        _img = np.array(Image.open(self.images[index]).convert('RGB')).astype(np.float32)
        return _img

    def _load_edge(self, index):
        _edge = np.load(self.edges[index]).astype(np.float32)
        return _edge

    def _load_semseg(self, index):
        # Note: We ignore the background class as other related works.
        _semseg = np.array(Image.open(self.semsegs[index])).astype(np.float32)
        _semseg[_semseg == 0] = 256
        _semseg = _semseg - 1
        return _semseg

    def _load_depth(self, index):
        _depth = np.load(self.depths[index])
        return _depth

    def _load_normals(self, index):
        _normals = np.load(self.normals[index])
        return _normals

    def _download(self):
        _fpath = os.path.join(MyPath.db_root_dir(), self.FILE)

        if os.path.isfile(_fpath):
            print('Files already downloaded')
            return
        else:
            print('Downloading from google drive')
            mkdir_if_missing(os.path.dirname(_fpath))
            download_file_from_google_drive(self.GOOGLE_DRIVE_ID, _fpath)

        # extract file
        cwd = os.getcwd()
        print('\nExtracting tar file')
        tar = tarfile.open(_fpath)
        os.chdir(MyPath.db_root_dir())
        tar.extractall()
        tar.close()
        os.chdir(cwd)
        print('Done!')

    def __str__(self):
        return 'NYUD Multitask (split=' + str(self.split) + ')'


def get_train_dataset(p, transforms, download=False):
    """ Return the train dataset """
    db_name = p['train_db_name']
    print('Preparing train loader for db: {}'.format(db_name))

    if db_name == 'NYUD':
        database = NYUD_MT(split='train', transform=transforms, do_edge='edge' in p.ALL_TASKS.NAMES,
                           do_semseg='semseg' in p.ALL_TASKS.NAMES, download=download,
                           do_normals='normals' in p.ALL_TASKS.NAMES,
                           do_depth='depth' in p.ALL_TASKS.NAMES, overfit=p['overfit'])
    elif db_name == 'PASCALContext':
        from utils.pascal_context import PASCALContext
        database = PASCALContext(split=['train'], transform=transforms, retname=True,
                                 do_semseg='semseg' in p.ALL_TASKS.NAMES, download=download,
                                 do_edge='edge' in p.ALL_TASKS.NAMES,
                                 do_normals='normals' in p.ALL_TASKS.NAMES,
                                 do_sal='sal' in p.ALL_TASKS.NAMES,
                                 do_human_parts='human_parts' in p.ALL_TASKS.NAMES,
                                 overfit=p['overfit'])

    else:
        raise NotImplemented("train_db_name: Choose among PASCALContext and NYUD")

    return database


def get_train_dataloader(p, batch_size, num_workers, dataset):
    """ Return the train dataloader """

    trainloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True,
                             num_workers=num_workers, collate_fn=collate_mil)
    return trainloader


def get_val_dataset(p, transforms, download=False):
    """ Return the validation dataset """

    db_name = p['val_db_name']
    print('Preparing val loader for db: {}'.format(db_name))

    if db_name == 'NYUD':
        database = NYUD_MT(split='val', transform=transforms, do_edge='edge' in p.TASKS.NAMES,
                           do_semseg='semseg' in p.TASKS.NAMES, download=download,
                           do_normals='normals' in p.TASKS.NAMES,
                           do_depth='depth' in p.TASKS.NAMES, overfit=p['overfit'])

    elif db_name == 'PASCALContext':
        from utils.pascal_context import PASCALContext
        database = PASCALContext(split=['val'], transform=transforms, retname=True,
                                 do_semseg='semseg' in p.TASKS.NAMES, download=download,
                                 do_edge='edge' in p.TASKS.NAMES,
                                 do_normals='normals' in p.TASKS.NAMES,
                                 do_sal='sal' in p.TASKS.NAMES,
                                 do_human_parts='human_parts' in p.TASKS.NAMES,
                                 overfit=p['overfit'])

    else:
        raise NotImplemented("test_db_name: Choose among PASCALContext and NYUD")

    return database


def get_val_dataloader(p, batch_size, num_workers, dataset):
    """ Return the validation dataloader """

    testloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, drop_last=False,
                            num_workers=num_workers)
    return testloader


""" 
    Loss functions 
"""


def get_loss(p, task=None):
    """ Return loss function for a specific task """

    if task == 'semseg' or task == 'human_parts':
        from utils.loss_functions import SoftMaxwithLoss
        criterion = SoftMaxwithLoss()


    elif task == 'depth':
        from utils.loss_functions import DepthLoss
        criterion = DepthLoss(p['depthloss'])

    elif task == 'semseg' or task == 'human_parts':
        from utils.loss_functions import SoftMaxwithLoss
        criterion = SoftMaxwithLoss()

    elif task == 'normals':
        from utils.loss_functions import NormalsLoss
        criterion = NormalsLoss(normalize=True, size_average=True, norm=p['normloss'])

    elif task == 'sal':
        from utils.loss_functions import BalancedCrossEntropyLoss
        criterion = BalancedCrossEntropyLoss(size_average=True)

    elif task == 'depth':
        from utils.loss_functions import DepthLoss
        criterion = DepthLoss(p['depthloss'])


    else:
        raise NotImplementedError('Undefined Loss: Choose a task among '
                                  'edge, semseg, human_parts, sal, depth, or normals')

    return criterion


def get_criterion(p):
    """ Return training criterion for a given setup """

    if p['setup'] == 'single_task':
        from utils.loss_schemes import SingleTaskLoss
        task = p.TASKS.NAMES[0]
        loss_ft = get_loss(p, task)
        return SingleTaskLoss(loss_ft, task)


    elif p['setup'] == 'multi_task':
        if p['loss_kwargs']['loss_scheme'] == 'baseline':  # Fixed weights
            from utils.loss_schemes import MultiTaskLoss
            loss_ft = torch.nn.ModuleDict({task: get_loss(p, task) for task in p.TASKS.NAMES})
            loss_weights = p['loss_kwargs']['loss_weights']
            if 'edge' in loss_weights.keys():
                loss_weights.pop("edge")
            return MultiTaskLoss(p.TASKS.NAMES, loss_ft, loss_weights)


        elif p['loss_kwargs']['loss_scheme'] == 'pad_net':  # Fixed weights but w/ deep supervision
            from utils.loss_schemes import PADNetLoss
            loss_ft = torch.nn.ModuleDict({task: get_loss(p, task) for task in p.ALL_TASKS.NAMES})
            loss_weights = p['loss_kwargs']['loss_weights']
            return PADNetLoss(p.TASKS.NAMES, p.AUXILARY_TASKS.NAMES, loss_ft, loss_weights)


        elif p['loss_kwargs']['loss_scheme'] == 'mti_net':  # Fixed weights but at multiple scales
            from utils.loss_schemes import MTINetLoss
            loss_ft = torch.nn.ModuleDict({task: get_loss(p, task) for task in set(p.ALL_TASKS.NAMES)})
            loss_weights = p['loss_kwargs']['loss_weights']
            return MTINetLoss(p.TASKS.NAMES, p.AUXILARY_TASKS.NAMES, loss_ft, loss_weights)
        else:
            raise NotImplementedError('Unknown loss scheme {}'.format(p['loss_kwargs']['loss_scheme']))

    else:
        raise NotImplementedError('Unknown setup {}'.format(p['setup']))


"""
    Optimizers and schedulers
"""


def get_optimizer(p, model):
    """ Return optimizer for a given model and setup """

    if p['model'] == 'cross_stitch':  # Custom learning rate for cross-stitch
        print('Optimizer uses custom scheme for cross-stitch nets')
        cross_stitch_params = [param for name, param in model.named_parameters() if 'cross_stitch' in name]
        single_task_params = [param for name, param in model.named_parameters() if not 'cross_stitch' in name]
        assert (p['optimizer'] == 'sgd')  # Adam seems to fail for cross-stitch nets
        optimizer = torch.optim.SGD([{'params': cross_stitch_params, 'lr': 100 * p['optimizer_kwargs']['lr']},
                                     {'params': single_task_params, 'lr': p['optimizer_kwargs']['lr']}],
                                    momentum=p['optimizer_kwargs']['momentum'],
                                    nesterov=p['optimizer_kwargs']['nesterov'],
                                    weight_decay=p['optimizer_kwargs']['weight_decay'])


    elif p['model'] == 'nddr_cnn':  # Custom learning rate for nddr-cnn
        print('Optimizer uses custom scheme for nddr-cnn nets')
        nddr_params = [param for name, param in model.named_parameters() if 'nddr' in name]
        single_task_params = [param for name, param in model.named_parameters() if not 'nddr' in name]
        assert (p['optimizer'] == 'sgd')  # Adam seems to fail for nddr-cnns
        optimizer = torch.optim.SGD([{'params': nddr_params, 'lr': 100 * p['optimizer_kwargs']['lr']},
                                     {'params': single_task_params, 'lr': p['optimizer_kwargs']['lr']}],
                                    momentum=p['optimizer_kwargs']['momentum'],
                                    nesterov=p['optimizer_kwargs']['nesterov'],
                                    weight_decay=p['optimizer_kwargs']['weight_decay'])


    else:  # Default. Same larning rate for all params
        print('Optimizer uses a single parameter group - (Default)')
        params = model.parameters()

        if p['optimizer'] == 'sgd':
            optimizer = torch.optim.SGD(params, **p['optimizer_kwargs'])

        elif p['optimizer'] == 'adam':
            optimizer = torch.optim.Adam(params, **p['optimizer_kwargs'])

        else:
            raise ValueError('Invalid optimizer {}'.format(p['optimizer']))

    return optimizer


def adjust_learning_rate(p, optimizer, epoch):
    """ Adjust the learning rate """

    lr = p['optimizer_kwargs']['lr']

    if p['scheduler'] == 'step':
        steps = np.sum(epoch > np.array(p['scheduler_kwargs']['lr_decay_epochs']))
        if steps > 0:
            lr = lr * (p['scheduler_kwargs']['lr_decay_rate'] ** steps)

    elif p['scheduler'] == 'poly':
        lambd = pow(1 - (epoch / p['epochs']), 0.9)
        lr = lr * lambd

    else:
        raise ValueError('Invalid learning rate schedule {}'.format(p['scheduler']))

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

    return lr

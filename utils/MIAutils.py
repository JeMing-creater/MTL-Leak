import os
import cv2
import torch
from torch import nn, optim
from torchvision import transforms, datasets
from torch.utils.data import DataLoader
import torch.nn.functional as F
import numpy as np
import torch.utils.data as Data
from torch.utils.data import random_split
from utils.custom_collate import collate_mil
import imageio
import json
import scipy.io as sio
from utils.utils import get_output, mkdir_if_missing

##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
## Segmentations-Leak: Membership Inference Attacks and Defenses in Semantic Image Segmentation
## by Yang He, Shadi Rahimian, Bernt Schiele and Mario Fritz, ECCV2020.
## The attacker (binary classification) implementation is based on third-party modules.
##
## Created by: Yang He - CISPA Helmholtz Center for Information Security
## Email: yang.he@cispa.saarland
##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++


"""Dilated ResNet"""
import math
import torch
import shutil
import torch.utils.model_zoo as model_zoo
import torch.nn as nn
from .customize import GlobalAvgPool2d

__all__ = ['ResNet', 'resnet18', 'resnet34', 'resnet50', 'resnet101',
           'resnet152', 'BasicBlock', 'Bottleneck', 'get_resnet']

model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
}


def conv3x3(in_planes, out_planes, stride=1):
    "3x3 convolution with padding"
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


class BasicBlock(nn.Module):
    """ResNet BasicBlock
    """
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, dilation=1, downsample=None, previous_dilation=1,
                 norm_layer=None):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=3, stride=stride,
                               padding=dilation, dilation=dilation, bias=False)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1,
                               padding=previous_dilation, dilation=previous_dilation, bias=False)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    """ResNet Bottleneck
    """
    # pylint: disable=unused-argument
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, dilation=1,
                 downsample=None, previous_dilation=1, norm_layer=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = norm_layer(planes)
        self.conv2 = nn.Conv2d(
            planes, planes, kernel_size=3, stride=stride,
            padding=dilation, dilation=dilation, bias=False)
        self.bn2 = norm_layer(planes)
        self.conv3 = nn.Conv2d(
            planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = norm_layer(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.dilation = dilation
        self.stride = stride

    def _sum_each(self, x, y):
        assert (len(x) == len(y))
        z = []
        for i in range(len(x)):
            z.append(x[i] + y[i])
        return z

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class ResNet(nn.Module):
    """Dilated Pre-trained ResNet Model, which preduces the stride of 8 featuremaps at conv5.

    Parameters
    ----------
    block : Block
        Class for the residual block. Options are BasicBlockV1, BottleneckV1.
    layers : list of int
        Numbers of layers in each block
    classes : int, default 1000
        Number of classification classes.
    dilated : bool, default False
        Applying dilation strategy to pretrained ResNet yielding a stride-8 model,
        typically used in Semantic Segmentation.
    norm_layer : object
        Normalization layer used in backbone network (default: :class:`mxnet.gluon.nn.BatchNorm`;
        for Synchronized Cross-GPU BachNormalization).

    Reference:

        - He, Kaiming, et al. "Deep residual learning for image recognition." Proceedings of the IEEE conference on computer vision and pattern recognition. 2016.

        - Yu, Fisher, and Vladlen Koltun. "Multi-scale context aggregation by dilated convolutions."
    """

    # pylint: disable=unused-variable
    def __init__(self, p, block, layers, input_channel=3, num_classes=1000, dilated=True, multi_grid=False,
                 deep_base=True, norm_layer=nn.BatchNorm2d, task_name=''):
        self.dataset = p['dataset']
        self.task_name = task_name
        self.inplanes = 128 if deep_base else 64
        super(ResNet, self).__init__()
        if deep_base:
            self.conv1 = nn.Sequential(
                nn.Conv2d(input_channel, 64, kernel_size=3, stride=2, padding=1, bias=False),
                norm_layer(64),
                nn.ReLU(inplace=True),
                nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=False),
                norm_layer(64),
                nn.ReLU(inplace=True),
                nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1, bias=False),
            )
        else:
            self.conv1 = nn.Conv2d(input_channel, 64, kernel_size=7, stride=2, padding=3,
                                   bias=False)
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0], norm_layer=norm_layer)
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2, norm_layer=norm_layer)
        if dilated:
            self.layer3 = self._make_layer(block, 256, layers[2], stride=1,
                                           dilation=2, norm_layer=norm_layer)
            if multi_grid:
                self.layer4 = self._make_layer(block, 512, layers[3], stride=1,
                                               dilation=4, norm_layer=norm_layer,
                                               multi_grid=True)
            else:
                self.layer4 = self._make_layer(block, 512, layers[3], stride=1,
                                               dilation=4, norm_layer=norm_layer)
        else:
            self.layer3 = self._make_layer(block, 256, layers[2], stride=1,
                                           norm_layer=norm_layer)
            self.layer4 = self._make_layer(block, 512, layers[3], stride=1,  # 2
                                           norm_layer=norm_layer)
        self.avgpool = GlobalAvgPool2d()
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, norm_layer):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1, dilation=1, norm_layer=None, multi_grid=False):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                norm_layer(planes * block.expansion),
            )

        layers = []
        multi_dilations = [4, 8, 16]
        if multi_grid:
            layers.append(block(self.inplanes, planes, stride, dilation=multi_dilations[0],
                                downsample=downsample, previous_dilation=dilation, norm_layer=norm_layer))
        elif dilation == 1 or dilation == 2:
            layers.append(block(self.inplanes, planes, stride, dilation=1,
                                downsample=downsample, previous_dilation=dilation, norm_layer=norm_layer))
        elif dilation == 4:
            layers.append(block(self.inplanes, planes, stride, dilation=2,
                                downsample=downsample, previous_dilation=dilation, norm_layer=norm_layer))
        else:
            raise RuntimeError("=> unknown dilation size: {}".format(dilation))

        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            if multi_grid:
                layers.append(block(self.inplanes, planes, dilation=multi_dilations[i],
                                    previous_dilation=dilation, norm_layer=norm_layer))
            else:
                layers.append(block(self.inplanes, planes, dilation=dilation, previous_dilation=dilation,
                                    norm_layer=norm_layer))

        return nn.Sequential(*layers)

    def forward(self, x):
        # if self.task_name == '':
        #     if self.dataset == 'nyud':
        #         x = torch.cat((x['semseg'], x['depth']), dim=1)
        #     else:
        #         d = torch.cat((x['semseg'], x['human_parts']), dim=1)
        #         d = torch.cat((d, x['sal']), dim=1)
        #         x = torch.cat((d, x['normals']), dim=1)
        # else:
        #     x = x
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x2 = self.layer4(x)

        x = self.avgpool(x2)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x  # , x2


def resnet18(p, pretrained=False, task_name='', **kwargs):
    """Constructs a ResNet-18 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(p, BasicBlock, [2, 2, 2, 2], task_name=task_name, **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet18']))
    return model


def resnet34(p, pretrained=False, task_name='', **kwargs):
    """Constructs a ResNet-34 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(p, BasicBlock, [3, 4, 6, 3], task_name=task_name, **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet34']))
    return model


def resnet50(p, pretrained=False, task_name='', **kwargs):
    """Constructs a ResNet-50 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(p, Bottleneck, [3, 4, 6, 3], task_name=task_name, **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet50']))
    return model


def resnet101(p, pretrained=False, task_name='', **kwargs):
    """Constructs a ResNet-101 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(p, Bottleneck, [3, 4, 23, 3], task_name=task_name, **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet101']))
    return model


def resnet152(p, pretrained=False, task_name='', **kwargs):
    """Constructs a ResNet-152 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(p, Bottleneck, [3, 8, 36, 3], task_name=task_name, **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet152']))
    return model


def get_resnet(p, arch, pretrained, task_name='', **kwargs):
    if arch == "resnet18":
        model = resnet18(p, pretrained, task_name, **kwargs)
    elif arch == "resnet34":
        model = resnet34(p, pretrained, task_name, **kwargs)
    elif arch == "resnet50":
        model = resnet50(p, pretrained, task_name, **kwargs)
    elif arch == "resnet101":
        model = resnet101(p, pretrained, task_name, **kwargs)
    elif arch == "resnet152":
        model = resnet152(p, pretrained, task_name, **kwargs)

    return model


def get_resnet_classification_model(p, arch, pretrained=False, task_name='', **kwargs):
    return get_resnet(p, arch, pretrained, task_name, **kwargs)


# 用以提供分类训练的基础模型Softmax
class Softmax_Model(nn.Module):

    def __init__(self, channel=41, size1=480, size2=640):

        super(Softmax_Model, self).__init__()
        self.channel = channel
        self.size1 = size1
        self.size2 = size2
        self.fc1 = nn.Linear(in_features=channel * size1 * size2, out_features=2)

        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.constant_(m.bias, 0.)

    def forward(self, x):
        # x1 = x['semseg']
        # # x1 = x1.reshape(-1, self.size1*self.size2)
        # # x1 = self.fc1(x1)
        # x2 = x['depth']
        # # x2 = x2.reshape(-1, self.size1*self.size2)
        # # x2 = self.fc1(x2)
        combined = torch.cat((x['semseg'], x['depth']), dim=1)
        # combined = torch.cat((x1, x2), dim=1)

        x = combined.reshape(8, self.channel * self.size1 * self.size2)
        result = self.fc1(x)

        return result


class Softmax_Model_for_singletask(nn.Module):
    def __init__(self, task_name='semseg', size1=480, size2=640):

        super(Softmax_Model_for_singletask, self).__init__()
        if task_name == 'semseg':
            self.channel = 40
        else:
            self.channel = 1
        self.size1 = size1
        self.size2 = size2
        self.fc1 = nn.Linear(in_features=self.channel * size1 * size2, out_features=2)

        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.constant_(m.bias, 0.)

    def forward(self, x):
        # x1 = x['semseg']
        # # x1 = x1.reshape(-1, self.size1*self.size2)
        # # x1 = self.fc1(x1)
        # x2 = x['depth']
        # # x2 = x2.reshape(-1, self.size1*self.size2)
        # # x2 = self.fc1(x2)

        # combined = torch.cat((x1, x2), dim=1)

        x = x.reshape(8, self.channel * self.size1 * self.size2)
        result = self.fc1(x)

        return result


class Softmax_with_conv_Model(nn.Module):
    def __init__(self, channel=41, size1=480, size2=640):
        super(Softmax_with_conv_Model, self).__init__()
        self.channel = channel
        self.size1 = size1
        self.size2 = size2
        self.conv1 = nn.Conv2d(in_channels=channel, out_channels=int(channel / 2), kernel_size=11)
        self.conv2 = nn.Conv2d(in_channels=int(channel / 2), out_channels=int(channel / 4), kernel_size=11)
        self.conv3 = nn.Conv2d(in_channels=int(channel / 4), out_channels=int(channel / 8), kernel_size=11)

        self.fc1 = nn.Linear(in_features=int(channel / 8) * int(size1 - 30) * int(size2 - 30), out_features=2)

        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.constant_(m.bias, 0.)

    def forward(self, x):
        combined = torch.cat((x['semseg'], x['depth']), dim=1)
        # 第一层卷积
        x = self.conv1(combined)
        # 第二层卷积
        x = self.conv2(x)
        # 第三层卷积
        x = self.conv3(x)
        # 拼接全链接
        # x = combined.reshape(8, int(self.channel/8) * (self.size1-30) * (self.size2-30))
        x = x.view(8, -1)
        result = self.fc1(x)
        return result


# Batch process the training data
def iterate_minibatches(inputs1, inputs2, targets, batch_size, shuffle=True, task_name=None):
    '''
    :param inputs: ndarray,such as (100,1,28,28)
    :param targets: ndarray,such as (100,10)
    :param batch_size: int, batch size of train data
    :param shuffle: decide whether to disrupt the data
    :return:
    '''
    assert len(inputs1) == len(inputs2) == len(targets)
    if shuffle:
        indices = np.arange(len(inputs1))
        np.random.shuffle(indices)

    start_idx = None
    for start_idx in range(0, len(inputs1) - batch_size + 1, batch_size):
        if shuffle:
            excerpt = indices[start_idx:start_idx + batch_size]
        else:
            excerpt = slice(start_idx, start_idx + batch_size)
        if task_name == None:
            yield inputs1[excerpt], inputs2[excerpt], targets[excerpt]
        elif task_name == 'semseg':
            yield inputs1[excerpt], targets[excerpt]
        else:
            yield inputs2[excerpt], targets[excerpt]

    if start_idx is not None and start_idx + batch_size < len(inputs1):
        excerpt = indices[start_idx + batch_size:] if shuffle else slice(start_idx + batch_size, len(inputs1))
        if task_name == None:
            yield inputs1[excerpt], inputs2[excerpt], targets[excerpt]
        elif task_name == 'semseg':
            yield inputs1[excerpt], targets[excerpt]
        else:
            yield inputs2[excerpt], targets[excerpt]


def iterate_minibatches_p(inputs1, inputs2, inputs3, inputs4, targets, batch_size, shuffle=True, task_name=None):
    '''
    :param inputs: ndarray,such as (100,1,28,28)
    :param targets: ndarray,such as (100,10)
    :param batch_size: int, batch size of train data
    :param shuffle: decide whether to disrupt the data
    :return:
    '''
    assert len(inputs1) == len(inputs2) == len(inputs3) == len(inputs4) == len(targets)
    if shuffle:
        indices = np.arange(len(inputs1))
        np.random.shuffle(indices)

    start_idx = None
    for start_idx in range(0, len(inputs1) - batch_size + 1, batch_size):
        if shuffle:
            excerpt = indices[start_idx:start_idx + batch_size]
        else:
            excerpt = slice(start_idx, start_idx + batch_size)
        if task_name == None:
            yield inputs1[excerpt], inputs2[excerpt], inputs3[excerpt], inputs4[excerpt], targets[excerpt]
        elif task_name == 'semseg':
            yield inputs1[excerpt], targets[excerpt]
        elif task_name == 'human_parts':
            yield inputs2[excerpt], targets[excerpt]
        elif task_name == 'sal':
            yield inputs3[excerpt], targets[excerpt]
        else:
            yield inputs4[excerpt], targets[excerpt]

    if start_idx is not None and start_idx + batch_size < len(inputs1):
        excerpt = indices[start_idx + batch_size:] if shuffle else slice(start_idx + batch_size, len(inputs1))
        if task_name == None:
            yield inputs1[excerpt], inputs2[excerpt], inputs3[excerpt], inputs4[excerpt], targets[excerpt]
        elif task_name == 'semseg':
            yield inputs1[excerpt], targets[excerpt]
        elif task_name == 'human_parts':
            yield inputs2[excerpt], targets[excerpt]
        elif task_name == 'sal':
            yield inputs3[excerpt], targets[excerpt]
        else:
            yield inputs4[excerpt], targets[excerpt]


# Batch process the training data
def iterate_minibatches_single(inputs, targets, batch_size, shuffle=True):
    '''
    :param inputs: ndarray,such as (100,1,28,28)
    :param targets: ndarray,such as (100,10)
    :param batch_size: int, batch size of train data
    :param shuffle: decide whether to disrupt the data
    :return:
    '''
    assert len(inputs) == len(targets)
    if shuffle:
        indices = np.arange(len(inputs))
        np.random.shuffle(indices)

    start_idx = None
    for start_idx in range(0, len(inputs) - batch_size + 1, batch_size):
        if shuffle:
            excerpt = indices[start_idx:start_idx + batch_size]
        else:
            excerpt = slice(start_idx, start_idx + batch_size)
        yield inputs[excerpt], targets[excerpt]

    if start_idx is not None and start_idx + batch_size < len(inputs):
        excerpt = indices[start_idx + batch_size:] if shuffle else slice(start_idx + batch_size, len(inputs))
        yield inputs[excerpt], targets[excerpt]


# 修改学习率
def mia_adjust_learning_rate(optimizer, epoch, lr):
    """Sets the learning rate to the initial LR decayed by 10 every 2 epochs"""
    lr *= (0.1 ** (epoch // 200))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


# 划分成员数据
def give_mia_data(train_dataset, val_dataset, batch_size=1, num_workers=0, shawdow_row=0.5, test_row=0.5):
    # 组合全部数据
    all_data = train_dataset + val_dataset
    target_size = int(shawdow_row * len(all_data))
    shawdow_size = len(all_data) - target_size
    target_data, shawdow_data = torch.utils.data.random_split(all_data,
                                                              [target_size, shawdow_size])
    # for target
    target_test_data = int(test_row * len(target_data))
    target_train_size = len(target_data) - target_test_data
    target_train_data, target_test_data = torch.utils.data.random_split(target_data,
                                                                        [target_train_size, target_test_data])
    # for shadow
    shawdow_test_data = int(test_row * len(shawdow_data))
    shawdow_train_size = len(shawdow_data) - shawdow_test_data
    shawdow_train_data, shawdow_test_data = torch.utils.data.random_split(shawdow_data,
                                                                          [shawdow_train_size, shawdow_test_data])
    # dataloader
    target_tarin_dataloader = DataLoader(target_train_data, batch_size=batch_size, shuffle=True, drop_last=True,
                                         num_workers=num_workers, collate_fn=collate_mil)
    target_val_dataloader = DataLoader(target_test_data, batch_size=batch_size, shuffle=True, drop_last=True,
                                       num_workers=num_workers, collate_fn=collate_mil)
    shawdow_tarin_dataloader = DataLoader(shawdow_train_data, batch_size=batch_size, shuffle=True, drop_last=True,
                                          num_workers=num_workers, collate_fn=collate_mil)
    shawdow_val_dataloader = DataLoader(shawdow_test_data, batch_size=batch_size, shuffle=True, drop_last=True,
                                        num_workers=num_workers, collate_fn=collate_mil)

    return target_tarin_dataloader, target_val_dataloader, shawdow_tarin_dataloader, shawdow_val_dataloader


# 数据保存
def give_data_save(p, ttrain, ttest, strain, stest, save_path='./datasets/MIAdata/', save_name='multi_task_baseline'):
    def save_data(data, save_path, save_name):
        size = data.batch_sampler.sampler.num_samples
        list = []
        for i, batch in enumerate(data):
            list.append(batch)
        if not os.path.exists(save_path):
            os.mkdir(save_path)
        np.save('{}/{}_pic.npy'.format(save_path, save_name), list)
        #     images = batch['image'].cuda(non_blocking=True)
        #     targets = {task: batch[task].cuda(non_blocking=True) for task in p.ALL_TASKS.NAMES}
        #     # meta保存
        #     meta = batch['meta']
        #
        #     img_channels = images.real.shape[1]
        #     img_size1 = images.real.shape[2]
        #     img_size2 = images.real.shape[3]
        #     t1_channels = targets['semseg'].real.shape[1]
        #     t1_size1 = targets['semseg'].real.shape[2]
        #     t1_size2 = targets['semseg'].real.shape[3]
        #     t2_channels = targets['depth'].real.shape[1]
        #     t2_size1 = targets['depth'].real.shape[2]
        #     t2_size2 = targets['depth'].real.shape[3]
        #     break
        # orginal_pic = np.zeros([size, img_channels, img_size1, img_size2])
        # label1 = np.zeros([size, t1_channels, t1_size1, t1_size2])
        # label2 = np.zeros([size, t2_channels, t2_size1, t2_size2])
        # with torch.no_grad():
        #     num = 0
        #     for i, batch in enumerate(data):
        #         images = batch['image'].cuda(non_blocking=True)
        #         targets = {task: batch[task].cuda(non_blocking=True) for task in p.ALL_TASKS.NAMES}
        #         for x in images:
        #             x = x.unsqueeze(0)
        #             orginal_pic[num, :, :, :] = x.cpu().data.numpy()
        #         for lb in targets['semseg']:
        #             lb = lb.unsqueeze(0)
        #             label1[num, :, :, :] = lb.cpu().data.numpy()
        #         for lb in targets['depth']:
        #             lb = lb.unsqueeze(0)
        #             label2[num, :, :, :] = lb.cpu().data.numpy()
        #         num += 1
        # np.save('{}/{}_pic.npy'.format(save_path, save_name), orginal_pic)
        # np.save('{}/{}_label1.npy'.format(save_path, save_name), label1)
        # np.save('{}/{}_label2.npy'.format(save_path, save_name), label1)

    save_data(ttrain, save_path=save_path, save_name=save_name + '_ttrain')
    save_data(ttest, save_path=save_path, save_name=save_name + '_ttest')
    save_data(strain, save_path=save_path, save_name=save_name + '_strain')
    save_data(stest, save_path=save_path, save_name=save_name + '_stest')
    print('all data save')


# 加载数据函数
def getData(data_path):
    def read_cam(path):
        with open(path, 'rb') as f:
            img2_array = np.load(f)
        return img2_array

    if 'pic' in data_path:
        print('read orginal_pic ...')
    elif 'label1' in data_path:
        print('read orginal_label1 ...')
    elif 'label2' in data_path:
        print('read orginal_label2 ...')
    else:
        print('data ...')
    data = read_cam(data_path).astype(np.float32)
    return data


# 提供训练替代模型的DataLoader
def givePreprocessingData(pic, batch_size=1):
    orginal_pic_transforms = torch.from_numpy(pic)

    dataset = Data.TensorDataset(orginal_pic_transforms)

    data = Data.DataLoader(
        dataset=dataset,  # 数据，封装进Data.TensorDataset()类的数据
        batch_size=batch_size,  # 每块的大小
        shuffle=True,  # 要不要打乱数据 (打乱比较好)
    )
    return data


def give_all_data(model_name):
    def get_data(model_name, data_name):
        pic_path = '{}_{}_pic.npy'.format(model_name, data_name)
        data = np.load(pic_path, allow_pickle=True).tolist()
        # pic_path = '{}_{}_pic.npy'.format(model_name,data_name)
        # label1_path = '{}_{}_label1.npy'.format(model_name,data_name)
        # label2_path = '{}_{}_label2.npy'.format(model_name, data_name)
        # data = givePreprocessingData(data, batch_size=batch_size)
        return data

    ttrain = get_data(model_name=model_name, data_name='ttrain')
    ttest = get_data(model_name=model_name, data_name='ttest')
    strain = get_data(model_name=model_name, data_name='strain')
    stest = get_data(model_name=model_name, data_name='stest')

    train_size = (len(ttrain) - 1) * ttrain[0]['image'].shape[0] + ttrain[-1]['image'].shape[0]
    shadow_size = (len(strain) - 1) * strain[0]['image'].shape[0] + strain[-1]['image'].shape[0]
    return ttrain, ttest, strain, stest, train_size, shadow_size


def del_files(dir_path):
    if os.path.isfile(dir_path):
        try:
            os.remove(dir_path)  # 这个可以删除单个文件，不能删除文件夹
        except BaseException as e:
            print(e)
    elif os.path.isdir(dir_path):
        file_lis = os.listdir(dir_path)
        for file_name in file_lis:
            # if file_name != 'wibot.log':
            tf = os.path.join(dir_path, file_name)
            del_files(tf)
    print('del ok')


@torch.no_grad()
def save_model_predictions(p, val_loader, model, accelerator):
    """ Save model predictions for all tasks """
    print('Save model predictions to {}'.format(p['save_dir']))
    model.eval()
    tasks = p.TASKS.NAMES
    save_dirs = {task: os.path.join(p['save_dir'], task) for task in tasks}
    for save_dir in save_dirs.values():
        mkdir_if_missing(save_dir)
    for task in p.TASKS.NAMES:
        if os.path.exists(save_dirs[task]):
            shutil.rmtree(save_dirs[task])
            os.makedirs(save_dirs[task])
        else:
            os.makedirs(save_dirs[task])
    # for ii, sample in enumerate(val_loader):
    # for ii, (img, label1, label2) in enumerate(val_loader):
    ii = 0
    for sample in val_loader:
        inputs, meta = sample['image'].to(accelerator.device), sample['meta']
        img_size = (inputs.size(2), inputs.size(3))
        output = model(inputs)

        for task in p.TASKS.NAMES:
            output_task = get_output(output[task], task).cpu().data.numpy()
            for jj in range(int(inputs.size()[0])):
                if len(sample[task][jj].unique()) == 1 and sample[task][jj].unique() == 255:
                    continue
                fname = meta['image'][jj]
                t1 = output_task[jj]
                t2 = meta['im_size'][1][jj]
                t3 = meta['im_size'][0][jj]
                t4 = p.TASKS.INFER_FLAGVALS[task]
                result = cv2.resize(output_task[jj], dsize=(meta['im_size'][1][jj], meta['im_size'][0][jj]),
                                    interpolation=p.TASKS.INFER_FLAGVALS[task])
                if task == 'depth':
                    sio.savemat(os.path.join(save_dirs[task], fname + '.mat'), {'depth': result})
                else:
                    imageio.imwrite(os.path.join(save_dirs[task], fname + '.png'), result.astype(np.uint8))

        ii += 0


def calculate_multi_task_performance(eval_dict, single_task_dict):
    assert (set(eval_dict.keys()) == set(single_task_dict.keys()))
    tasks = eval_dict.keys()
    num_tasks = len(tasks)
    mtl_performance = 0.0

    for task in tasks:
        mtl = eval_dict[task]
        stl = single_task_dict[task]

        if task == 'depth':  # rmse lower is better
            mtl_performance -= (mtl['rmse'] - stl['rmse']) / stl['rmse']

        elif task in ['semseg', 'sal', 'human_parts']:  # mIoU higher is better
            mtl_performance += (mtl['mIoU'] - stl['mIoU']) / stl['mIoU']

        elif task == 'normals':  # mean error lower is better
            mtl_performance -= (mtl['mean'] - stl['mean']) / stl['mean']

        elif task == 'edge':  # odsF higher is better
            mtl_performance += (mtl['odsF'] - stl['odsF']) / stl['odsF']

        else:
            raise NotImplementedError

    return mtl_performance / num_tasks


def eval_all_results(p, definition_db):
    save_dir = p['save_dir']
    results = {}
    if 'semseg' in p.TASKS.NAMES:
        from utils.eval_semseg import eval_semseg_predictions
        results['semseg'] = eval_semseg_predictions(database=p['val_db_name'],
                                                    save_dir=save_dir, overfit=p.overfit, definition_db=definition_db)

    if 'depth' in p.TASKS.NAMES:
        from utils.eval_depth import eval_depth_predictions
        results['depth'] = eval_depth_predictions(database=p['val_db_name'],
                                                  save_dir=save_dir, overfit=p.overfit, definition_db=definition_db)

    if 'human_parts' in p.TASKS.NAMES:
        from utils.eval_human_parts import eval_human_parts_predictions
        results['human_parts'] = eval_human_parts_predictions(database=p['val_db_name'],
                                                              save_dir=save_dir, overfit=p.overfit,
                                                              definition_db=definition_db)

    if 'sal' in p.TASKS.NAMES:
        from utils.eval_sal import eval_sal_predictions
        results['sal'] = eval_sal_predictions(database=p['val_db_name'],
                                              save_dir=save_dir, overfit=p.overfit, definition_db=definition_db)

    if 'normals' in p.TASKS.NAMES:
        from utils.eval_normals import eval_normals_predictions
        results['normals'] = eval_normals_predictions(database=p['val_db_name'],
                                                      save_dir=save_dir, overfit=p.overfit, definition_db=definition_db)

    if p['setup'] == 'multi_task':  # Perform the multi-task performance evaluation
        single_task_test_dict = {}
        for task, test_dict in p.TASKS.SINGLE_TASK_TEST_DICT.items():
            with open(test_dict, 'r') as f_:
                single_task_test_dict[task] = json.load(f_)
        results['multi_task_performance'] = calculate_multi_task_performance(results, single_task_test_dict)
        print('Multi-task learning performance on test set is %.2f' % (100 * results['multi_task_performance']))
    return results


def give_attack_data(p, target_model, shadow_model, dataset, model_name, accelerator, save_path='./datasets/MIAdata',
                     keep_data_row=0.8):
    ttrain, ttest, strain, stest = dataset
    target_data = (ttrain, ttest)
    shadow_data = (strain, stest)

    def give_label_data(p, model, dataset, accelerator, keep_data_row=0.8):
        train, test = dataset
        if p['dataset'] == 'nyud':
            attack_x1, attack_x2, attack_y = [], [], []
            model.eval()
            for batch in train:
                images = batch['image'].to(accelerator.device)
                output = model(images)
                new_dict = {task: output[task].detach().cpu().numpy() for task in p.ALL_TASKS.NAMES}
                attack_x1.append(new_dict['semseg'])
                attack_x2.append(new_dict['depth'])
                attack_y.append(np.ones(batch['image'].shape[0]))
            for batch in test:
                images = batch['image'].to(accelerator.device)
                output = model(images)
                new_dict = {task: output[task].detach().cpu().numpy() for task in p.ALL_TASKS.NAMES}
                attack_x1.append(new_dict['semseg'])
                attack_x2.append(new_dict['depth'])
                attack_y.append(np.zeros(batch['image'].shape[0]))
            return attack_x1, attack_x2, attack_y
        else:
            attack_x1, attack_x2, attack_x3, attack_x4, attack_y = [], [], [], [], []
            model.eval()
            keep_data_num = int(keep_data_row * len(train))
            num = 0
            for batch in train:
                num += 1
                if num >= keep_data_num:
                    num = 0
                    break
                images = batch['image'].to(accelerator.device)
                output = model(images)
                new_dict = {task: output[task].detach().cpu().numpy() for task in p.ALL_TASKS.NAMES}
                attack_x1.append(new_dict['semseg'])
                attack_x2.append(new_dict['human_parts'])
                attack_x3.append(new_dict['sal'])
                attack_x4.append(new_dict['normals'])
                attack_y.append(np.ones(batch['image'].shape[0]))
            keep_data_num = int(keep_data_row * len(test))
            for batch in test:
                num += 1
                if num > keep_data_num:
                    num = 0
                    break
                images = batch['image'].to(accelerator.device)
                output = model(images)
                new_dict = {task: output[task].detach().cpu().numpy() for task in p.ALL_TASKS.NAMES}
                attack_x1.append(new_dict['semseg'])
                attack_x2.append(new_dict['human_parts'])
                attack_x3.append(new_dict['sal'])
                attack_x4.append(new_dict['normals'])
                attack_y.append(np.zeros(batch['image'].shape[0]))
            print('keep data is {}'.format(keep_data_num))
            return attack_x1, attack_x2, attack_x3, attack_x4, attack_y

    if p['dataset'] == 'nyud':
        target_attack_x1, target_attack_x2, target_attack_y = give_label_data(p=p, model=target_model,
                                                                              accelerator=accelerator,
                                                                              dataset=target_data,
                                                                              keep_data_row=keep_data_row)
        target_attack_x1 = np.vstack(target_attack_x1).astype('float32')
        target_attack_x2 = np.vstack(target_attack_x2).astype('float32')
        target_attack_y = np.concatenate(target_attack_y).astype('int32')
        np.save('{}/{}_target_attack_x1.npy'.format(save_path, model_name), target_attack_x1)
        np.save('{}/{}_target_attack_x2.npy'.format(save_path, model_name), target_attack_x2)
        np.save('{}/{}_target_attacky.npy'.format(save_path, model_name), target_attack_y)
        print('target_data preprocessing...')
        shadow_attack_x1, shadow_attack_x2, shadow_attack_y = give_label_data(p=p, model=shadow_model,
                                                                              accelerator=accelerator,
                                                                              dataset=shadow_data,
                                                                              keep_data_row=keep_data_row)
        shadow_attack_x1 = np.vstack(shadow_attack_x1).astype('float32')
        shadow_attack_x2 = np.vstack(shadow_attack_x2).astype('float32')
        shadow_attack_y = np.concatenate(shadow_attack_y).astype('int32')
        np.save('{}/{}_shadow_attack_x1.npy'.format(save_path, model_name), shadow_attack_x1)
        np.save('{}/{}_shadow_attack_x2.npy'.format(save_path, model_name), shadow_attack_x2)
        np.save('{}/{}_shadow_attacky.npy'.format(save_path, model_name), shadow_attack_y)
        print('shadow_data preprocessing...')
        return target_attack_x1, target_attack_x2, target_attack_y, shadow_attack_x1, shadow_attack_x2, shadow_attack_y

    else:
        target_attack_x1, target_attack_x2, target_attack_x3, target_attack_x4, target_attack_y = give_label_data(p=p,
                                                                                                                  model=target_model,
                                                                                                                  dataset=target_data,
                                                                                                                  keep_data_row=keep_data_row)
        target_attack_x1 = np.vstack(target_attack_x1).astype('float32')
        target_attack_x2 = np.vstack(target_attack_x2).astype('float32')
        target_attack_x3 = np.vstack(target_attack_x3).astype('float32')
        target_attack_x4 = np.vstack(target_attack_x4).astype('float32')
        target_attack_y = np.concatenate(target_attack_y).astype('int32')
        np.save('{}/{}_target_attack_x1.npy'.format(save_path, model_name), target_attack_x1)
        np.save('{}/{}_target_attack_x2.npy'.format(save_path, model_name), target_attack_x2)
        np.save('{}/{}_target_attack_x3.npy'.format(save_path, model_name), target_attack_x3)
        np.save('{}/{}_target_attack_x4.npy'.format(save_path, model_name), target_attack_x4)
        np.save('{}/{}_target_attacky.npy'.format(save_path, model_name), target_attack_y)
        print('target_data preprocessing...')
        shadow_attack_x1, shadow_attack_x2, shadow_attack_x3, shadow_attack_x4, shadow_attack_y = give_label_data(p=p,
                                                                                                                  model=shadow_model,
                                                                                                                  dataset=shadow_data,
                                                                                                                  keep_data_row=keep_data_row)
        shadow_attack_x1 = np.vstack(shadow_attack_x1).astype('float32')
        shadow_attack_x2 = np.vstack(shadow_attack_x2).astype('float32')
        shadow_attack_x3 = np.vstack(shadow_attack_x3).astype('float32')
        shadow_attack_x4 = np.vstack(shadow_attack_x4).astype('float32')
        shadow_attack_y = np.concatenate(shadow_attack_y).astype('int32')
        np.save('{}/{}_shadow_attack_x1.npy'.format(save_path, model_name), shadow_attack_x1)
        np.save('{}/{}_shadow_attack_x2.npy'.format(save_path, model_name), shadow_attack_x2)
        np.save('{}/{}_shadow_attack_x3.npy'.format(save_path, model_name), shadow_attack_x3)
        np.save('{}/{}_shadow_attack_x4.npy'.format(save_path, model_name), shadow_attack_x4)
        np.save('{}/{}_shadow_attacky.npy'.format(save_path, model_name), shadow_attack_y)
        print('shadow_data preprocessing...')
        return target_attack_x1, target_attack_x2, target_attack_x3, target_attack_x4, target_attack_y, shadow_attack_x1, shadow_attack_x2, shadow_attack_x3, shadow_attack_x4, shadow_attack_y
    # return target_attack_x1, target_attack_x2, target_attack_y


def give_single_attack_data(p, target_model, shadow_model, dataset, model_name, task_name='semseg',
                            save_path='./datasets/MIAdata'):
    ttrain, ttest, strain, stest = dataset
    target_data = (ttrain, ttest)
    shadow_data = (strain, stest)

    def give_label_data(p, model, dataset, task_name):
        train, test = dataset
        attack_x1, attack_y = [], []
        model.eval()
        for batch in train:
            images = batch['image'].cuda(non_blocking=True)
            output = model.cuda()(images)
            new_dict = {task: output[task].detach().cpu().numpy() for task in p.ALL_TASKS.NAMES}
            attack_x1.append(new_dict[task_name])
            attack_y.append(np.ones(batch['image'].shape[0]))
        for batch in test:
            images = batch['image'].cuda(non_blocking=True)
            output = model.cuda()(images)
            new_dict = {task: output[task].detach().cpu().numpy() for task in p.ALL_TASKS.NAMES}
            attack_x1.append(new_dict[task_name])
            attack_y.append(np.zeros(batch['image'].shape[0]))
        return attack_x1, attack_y

    target_attack_x1, target_attack_y = give_label_data(p=p, model=target_model, dataset=target_data,
                                                        task_name=task_name)
    target_attack_x1 = np.vstack(target_attack_x1).astype('float32')
    target_attack_y = np.concatenate(target_attack_y).astype('int32')
    np.save('{}/{}_target_attack_x1.npy'.format(save_path, model_name), target_attack_x1)
    np.save('{}/{}_target_attacky.npy'.format(save_path, model_name), target_attack_y)
    print('target_data preprocessing...')
    shadow_attack_x1, shadow_attack_y = give_label_data(p=p, model=shadow_model, dataset=shadow_data,
                                                        task_name=task_name)
    shadow_attack_x1 = np.vstack(shadow_attack_x1).astype('float32')
    shadow_attack_y = np.concatenate(shadow_attack_y).astype('int32')
    np.save('{}/{}_shadow_attack_x1.npy'.format(save_path, model_name), shadow_attack_x1)
    np.save('{}/{}_shadow_attacky.npy'.format(save_path, model_name), shadow_attack_y)
    print('shadow_data preprocessing...')
    return target_attack_x1, target_attack_y, shadow_attack_x1, shadow_attack_y
    # return target_attack_x1, target_attack_x2, target_attack_y


def give_single_attack_data_t(p, target_model, shadow_model, dataset, model_name, task_name='semseg',
                              save_path='./datasets/MIAdata', keep_data_row=0.8):
    ttrain, ttest, strain, stest = dataset
    target_data = (ttrain, ttest)
    shadow_data = (strain, stest)

    def give_label_data(p, model, dataset, task_name, keep_data_row):
        train, test = dataset
        attack_x1, attack_y = [], []
        model.eval()
        keep_data = int(keep_data_row * len(test))
        print('keep data num is {}'.format(keep_data))
        num = 0
        keep_data = int(keep_data_row * len(train))
        for batch in train:
            if num > keep_data:
                num = 0
                break
            num += 1
            images = batch['image'].cuda(non_blocking=True)
            output = model.cuda()(images)
            new_dict = {task: output[task].detach().cpu().numpy() for task in p.ALL_TASKS.NAMES}
            attack_x1.append(new_dict[task_name])
            attack_y.append(np.ones(batch['image'].shape[0]))
        keep_data = int(keep_data_row * len(test))
        for batch in test:
            if num > keep_data:
                num = 0
                break
            num += 1
            images = batch['image'].cuda(non_blocking=True)
            output = model.cuda()(images)
            new_dict = {task: output[task].detach().cpu().numpy() for task in p.ALL_TASKS.NAMES}
            attack_x1.append(new_dict[task_name])
            attack_y.append(np.zeros(batch['image'].shape[0]))
        return attack_x1, attack_y

    target_attack_x1, target_attack_y = give_label_data(p=p, model=target_model, dataset=target_data,
                                                        task_name=task_name, keep_data_row=keep_data_row)
    target_attack_x1 = np.vstack(target_attack_x1).astype('float32')
    target_attack_y = np.concatenate(target_attack_y).astype('int32')
    np.save('{}/{}_target_attack_x1.npy'.format(save_path, model_name), target_attack_x1)
    np.save('{}/{}_target_attacky.npy'.format(save_path, model_name), target_attack_y)
    print('target_data preprocessing...')
    shadow_attack_x1, shadow_attack_y = give_label_data(p=p, model=shadow_model, dataset=shadow_data,
                                                        task_name=task_name, keep_data_row=keep_data_row)
    shadow_attack_x1 = np.vstack(shadow_attack_x1).astype('float32')
    shadow_attack_y = np.concatenate(shadow_attack_y).astype('int32')
    np.save('{}/{}_shadow_attack_x1.npy'.format(save_path, model_name), shadow_attack_x1)
    np.save('{}/{}_shadow_attacky.npy'.format(save_path, model_name), shadow_attack_y)
    print('shadow_data preprocessing...')
    return target_attack_x1, target_attack_y, shadow_attack_x1, shadow_attack_y


def giveLabeledData(model_name, save_path='./datasets/MIAdata'):
    def give_data(attack_x1_path, attack_x2_path, attack_y_path):
        def getData(data_path):
            def read_cam(path):
                with open(path, 'rb') as f:
                    img2_array = np.load(f)
                return img2_array

            data = read_cam(data_path).astype(np.float32)
            return data

        attack_x1_transforms = getData(attack_x1_path).astype(np.float32)
        attack_x2_transforms = getData(attack_x2_path).astype(np.float32)
        attack_y_transforms = getData(attack_y_path).astype('int32')
        return attack_x1_transforms, attack_x2_transforms, attack_y_transforms

    tx1 = save_path + '/' + model_name + '_target_attack_x1.npy'
    tx2 = save_path + '/' + model_name + '_target_attack_x2.npy'
    ty = save_path + '/' + model_name + '_target_attacky.npy'
    sx1 = save_path + '/' + model_name + '_shadow_attack_x1.npy'
    sx2 = save_path + '/' + model_name + '_shadow_attack_x2.npy'
    sy = save_path + '/' + model_name + '_shadow_attacky.npy'

    ttx1, ttx2, tty = give_data(tx1, tx2, ty)
    print('target_data have loaded')
    stx1, stx2, sty = give_data(sx1, sx2, sy)
    print('shadow_data have loaded')
    return ttx1, ttx2, tty, stx1, stx2, sty


def giveLabeledData_single(model_name, save_path='./datasets/MIAdata'):
    def give_data(attack_x1_path, attack_y_path):
        def getData(data_path):
            def read_cam(path):
                with open(path, 'rb') as f:
                    img2_array = np.load(f)
                return img2_array

            data = read_cam(data_path).astype(np.float32)
            return data

        attack_x1_transforms = getData(attack_x1_path).astype(np.float32)
        attack_y_transforms = getData(attack_y_path).astype('int32')
        return attack_x1_transforms, attack_y_transforms

    tx1 = save_path + '/' + model_name + '_target_attack_x1.npy'
    ty = save_path + '/' + model_name + '_target_attacky.npy'
    sx1 = save_path + '/' + model_name + '_shadow_attack_x1.npy'
    sy = save_path + '/' + model_name + '_shadow_attacky.npy'

    ttx1, tty = give_data(tx1, ty)
    print('target_data have loaded')
    stx1, sty = give_data(sx1, sy)
    print('shadow_data have loaded')
    return ttx1, tty, stx1, sty


def get_log(log_path, log):
    with open(log_path, 'a') as f:
        f.write(log)
        f.write('\n')
        f.close()

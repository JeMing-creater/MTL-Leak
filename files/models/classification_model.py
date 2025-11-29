import torch
import torch.nn.functional
from functools import partial
import torch.nn.functional as F

from torch import nn, einsum
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
import math

import warnings
from math import sqrt

# Difference module
import torch.nn as nn

class PatchPartition(nn.Module):
    def __init__(self, channels):
        super(PatchPartition, self).__init__()
        self.positional_encoding = nn.Conv2d(channels, channels, kernel_size=3, padding=1, groups=channels,
                                                         bias=False)

    def forward(self, x):
        x = self.positional_encoding(x)
        return x

class LineConv(nn.Module):
    def __init__(self, channels):
        super(LineConv, self).__init__()
        expansion = 4
        self.line_conv_0 = nn.Conv2d(channels, channels * expansion, kernel_size=1, bias=False)
        self.act = nn.GELU()
        self.line_conv_1 = nn.Conv2d(channels * expansion, channels, kernel_size=1, bias=False)

    def forward(self, x):
        x = self.line_conv_0(x)
        x = self.act(x)
        x = self.line_conv_1(x)
        return x

class LocalRepresentationsCongregation(nn.Module):
    def __init__(self, channels):
        super(LocalRepresentationsCongregation, self).__init__()
        self.bn1 = nn.BatchNorm2d(channels)
        self.pointwise_conv_0 = nn.Conv2d(channels, channels, kernel_size=1, bias=False)
        self.depthwise_conv = nn.Conv2d(channels, channels, padding=1, kernel_size=3, groups=channels, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)
        self.pointwise_conv_1 = nn.Conv2d(channels, channels, kernel_size=1, bias=False)

    def forward(self, x):
        x = self.bn1(x)
        x = self.pointwise_conv_0(x)
        x = self.depthwise_conv(x)
        x = self.bn2(x)
        x = self.pointwise_conv_1(x)
        return x


class GlobalSparseTransformer(nn.Module):
    def __init__(self, channels, r, heads):
        super(GlobalSparseTransformer, self).__init__()
        self.head_dim = channels // heads
        self.scale = self.head_dim ** -0.5
        self.num_heads = heads
        self.sparse_sampler = nn.AvgPool2d(kernel_size=1, stride=r)
        # qkv
        self.qkv = nn.Conv2d(channels, channels * 3, kernel_size=1, bias=False)

    def forward(self, x):
        x = self.sparse_sampler(x)
        B, C, H, W = x.shape
        q, k, v = self.qkv(x).view(B, self.num_heads, -1, H * W).split([self.head_dim, self.head_dim, self.head_dim],
                                                                       dim=2)
        # 计算特征图 attention map
        attn = (q.transpose(-2, -1) @ k).softmax(-1)
        # value和特征图进行计算，得出全局注意力的结果
        x = (v @ attn.transpose(-2, -1)).view(B, -1, H, W)
        # print(x.shape)
        return x


class LocalReverseDiffusion(nn.Module):
    def __init__(self, channels, r):
        super(LocalReverseDiffusion, self).__init__()
        self.norm = nn.GroupNorm(num_groups=1, num_channels=channels)
        self.conv_trans = nn.ConvTranspose2d(channels,
                                             channels,
                                             kernel_size=r,
                                             stride=r,
                                             groups=channels)
        self.pointwise_conv = nn.Conv2d(channels, channels, kernel_size=1, bias=False)

    def forward(self, x):
        x = self.conv_trans(x)
        x = self.norm(x)
        x = self.pointwise_conv(x)
        return x


class Block(nn.Module):
    def __init__(self, channels, r, heads):
        super(Block, self).__init__()

        self.patch1 = PatchPartition(channels)
        self.LocalRC = LocalRepresentationsCongregation(channels)
        self.LineConv1 = LineConv(channels)
        self.patch2 = PatchPartition(channels)
        self.GlobalST = GlobalSparseTransformer(channels, r, heads)
        self.LocalRD = LocalReverseDiffusion(channels, r)
        self.LineConv2 = LineConv(channels)

    def forward(self, x):
        x = self.patch1(x) + x
        x = self.LocalRC(x) + x
        x = self.LineConv1(x) + x
        x = self.patch2(x) + x
        x = self.LocalRD(self.GlobalST(x)) + x
        x = self.LineConv2(x) + x
        return x

class DepthwiseConvLayer(nn.Module):
    def __init__(self, dim_in, dim_out, r):
        super(DepthwiseConvLayer, self).__init__()
        self.depth_wise = nn.Conv2d(dim_in,
                                    dim_out,
                                    kernel_size=r,
                                    stride=r)
        self.norm = nn.GroupNorm(num_groups=1, num_channels=dim_out)

    def forward(self, x):
        x = self.depth_wise(x)
        x = self.norm(x)
        return x

class Encoder(nn.Module):
    def __init__(self, in_channels=1, embed_dim=384, channels=(48, 96, 240),
                 blocks=(1, 2, 3, 2), heads=(1, 2, 4, 8), r=(4, 2, 2, 1), distillation=False, dropout=0.3):
        super(Encoder, self).__init__()
        self.distillation = distillation
        self.DWconv1 = DepthwiseConvLayer(dim_in=in_channels, dim_out=channels[0], r=4)
        self.DWconv2 = DepthwiseConvLayer(dim_in=channels[0], dim_out=channels[1], r=2)
        self.DWconv3 = DepthwiseConvLayer(dim_in=channels[1], dim_out=channels[2], r=2)
        self.DWconv4 = DepthwiseConvLayer(dim_in=channels[2], dim_out=embed_dim, r=2)
        block = []
        for _ in range(blocks[0]):
            block.append(Block(channels=channels[0], r=r[0], heads=heads[0]))
        self.block1 = nn.Sequential(*block)
        block = []
        for _ in range(blocks[1]):
            block.append(Block(channels=channels[1], r=r[1], heads=heads[1]))
        self.block2 = nn.Sequential(*block)
        block = []
        for _ in range(blocks[2]):
            block.append(Block(channels=channels[2], r=r[2], heads=heads[2]))
        self.block3 = nn.Sequential(*block)
        block = []
        for _ in range(blocks[3]):
            block.append(Block(channels=embed_dim, r=r[3], heads=heads[3]))
        self.block4 = nn.Sequential(*block)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        hidden_states_out = []
        x = self.DWconv1(x)
        x = self.block1(x)
        hidden_states_out.append(x)
        x = self.DWconv2(x)
        x = self.block2(x)
        hidden_states_out.append(x)
        x = self.DWconv3(x)
        x = self.block3(x)
        hidden_states_out.append(x)
        x = self.DWconv4(x)
        x = self.block4(x)
        B, C, W, H = x.shape
        x = x.flatten(2).transpose(-1, -2)
        x = self.dropout(x)

        return x

class ClassFormer(nn.Module):
    def __init__(self, in_channels=1,out_channels=2, embed_dim=384, channels=(48, 96, 240),
                 blocks=(1, 2, 3, 2), heads=(1, 2, 4, 8), r=(4, 2, 2, 1), distillation=False, dropout=0.3):
        super().__init__()
        self.e = Encoder(in_channels=in_channels,embed_dim=embed_dim, channels=channels,
                 blocks=blocks, heads=heads, r=r, distillation=distillation, dropout=dropout )
        self.fc1 = nn.Sequential(
            # nn.Linear(in_features=32 * 5 * 5, out_features=n_hidden),         # Mnist
            nn.Linear(in_features=115200, out_features=50),  # cifar 10 or cifar 100
            nn.Tanh()
        )

        self.fc2 = nn.Sequential(
            nn.Linear(in_features=50, out_features=out_channels)
        )

    def forward(self, x):

        x_features = self.e(x)
        x = x_features.reshape(x_features.size(0), -1)
        x = self.fc1(x)
        x = self.fc2(x)
        # x = self.d(x1_features, x2_features)
        return x


if __name__ == '__main__':
    # device = 'cuda:0'
    # x1 = torch.rand(1, 40, 480, 640)
    x = torch.rand(1, 3, 480, 640)
    model = ClassFormer()
    out = model(x)
    print(out.shape)

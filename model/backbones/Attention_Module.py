# -*- coding: utf-8 -*-#

# -------------------------------------------------------------------------------
# Name:         Attention_Module
# Author:       LPT
# Email:        lpt2820447@163.com
# Date:         2021/11/17 16:43
# Description:
# -------------------------------------------------------------------------------

import torch
import numpy as np
import torch.nn as nn
import math
import sys
from IPython import embed

class RGA(object):
    def __init__(self, img_size=(256, 128), stride_size=(16, 16), patch_size=(16, 16), embed_dim=768, divide_length=None, **kwargs):
        self.name = 'Relation_aware'
        print("==========Use 'RGA' Attention Model for Transformer!==========")
        self.img_size = img_size
        self.stride_size = stride_size
        self.embed_dim = embed_dim
        self.divide_length = divide_length
        x = (img_size[0] - patch_size[0]) // stride_size[0] + 1
        y = (img_size[1] - patch_size[1]) // stride_size[1] + 1
        self.num_patch = x * y
        self.feature_size = (x, y)
        if divide_length is not None:
            self.Spatial = nn.ModuleList([])
            feature_size = ((x // divide_length), y)
            for i in range(self.divide_length):
                if i == (divide_length - 1):
                    feature_size =  (x - (x // divide_length) * i, y)
                self.Spatial.append(Spatial_Attention_RGA(feature_size=feature_size))
        else:
            self.Spatial = Spatial_Attention_RGA(feature_size=self.feature_size)
        self.Channel = Channel_Attention_RGA(embed_dim=embed_dim, num_vector=self.num_patch)

class Spatial_Attention_RGA(nn.Module):
    """ Image to Patch Embedding with overlapping patches
    """
    def __init__(self, embed_dim=768, down_ratio=4, feature_size=(16, 8)):
        super().__init__()
        self.embed_dim = embed_dim
        self.feature_size = feature_size
        self.num_patches = feature_size[0] * feature_size[1]
        self.inter_channel = 32
        self.theta_spatial = nn.Sequential(
            nn.Conv2d(in_channels=embed_dim, out_channels=self.inter_channel,
                      kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(self.inter_channel),
            nn.ReLU()
        )

        self.phi_spatial = nn.Sequential(
            nn.Conv2d(in_channels=embed_dim, out_channels=self.inter_channel,
                      kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(self.inter_channel),
            nn.ReLU()
        )
        self.inter_spatial = 128 // 8
        self.gg_spatial = nn.Sequential(
            nn.Conv2d(in_channels=self.num_patches * 2, out_channels=self.inter_spatial,
                      kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(self.inter_spatial),
            nn.ReLU()
        )

        self.gx_spatial = nn.Sequential(
            nn.Conv2d(in_channels=embed_dim, out_channels=self.inter_channel,
                      kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(self.inter_channel),
            nn.ReLU()
        )

        num_channel_s = 1 + self.inter_spatial
        self.W_spatial = nn.Sequential(
            nn.Conv2d(in_channels=num_channel_s, out_channels=num_channel_s // down_ratio,
                      kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(num_channel_s // down_ratio),
            nn.ReLU(),
            nn.Conv2d(in_channels=num_channel_s // down_ratio, out_channels=1,
                      kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(1)
        )

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.InstanceNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, x):
        B, s, C = x.shape
        tx = x.transpose(1, 2)
        tx = tx.view(B, C, -1, self.feature_size[-1])
        B, C, h, w = tx.shape
        theta_xs = self.theta_spatial(tx)    # 8,32,16,8
        phi_xs = self.phi_spatial(tx)        # 8,32,16,8

        theta_xs = theta_xs.view(B, self.inter_channel, -1) # 8,32,128
        theta_xs = theta_xs.transpose(1,2)  # 8,128,32

        phi_xs = phi_xs.view(B, self.inter_channel, -1) # 8,32,128

        Gs = torch.matmul(theta_xs, phi_xs) # 8,128,128
        Gs_in = Gs.transpose(1, 2).view(B, h * w, h, w) # 8,128,16,8
        Gs_out = Gs.view(B, h * w, h, w)    # 8,128,16,8
        Gs_joint = torch.cat((Gs_in, Gs_out), 1)    # 8,256,16,8
        Gs_joint = self.gg_spatial(Gs_joint)    # 8,16,16,8

        g_xs = self.gx_spatial(tx)   # 8,16,16,8
        g_xs = torch.mean(g_xs, dim=1, keepdim=True)    # 8,1,16,8
        ys = torch.cat((g_xs, Gs_joint), dim=1) # 8,17,16,8

        W_ys = self.W_spatial(ys)
        res = torch.sigmoid(W_ys.expand_as(tx)).view(B, C, s).transpose(1, 2)
        return res

class Channel_Attention_RGA(nn.Module):
    """ Image to Patch Embedding with overlapping patches
    input must be a vector(b, n, dim), b is batch, n is the num of data, dim is dimension of data
    in_chans: numer of input data
    embed_dim: input vector's dim
    down_ratio: the sampling ratio of Channel down
    """
    def __init__(self, num_vector=128, embed_dim=768, down_ratio=4):

        super().__init__()
        self.in_channel = embed_dim
        self.in_spatial = num_vector
        self.inter_spatial = self.in_spatial // down_ratio
        self.inter_channel = self.in_channel // down_ratio
        self.theta_channel = nn.Sequential(
            nn.Conv2d(in_channels=self.in_spatial, out_channels=self.inter_spatial,
                      kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(self.inter_spatial),
            nn.ReLU()
        )
        self.phi_channel = nn.Sequential(
            nn.Conv2d(in_channels=self.in_spatial, out_channels=self.inter_spatial,
                      kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(self.inter_spatial),
            nn.ReLU()
        )
        self.gg_channel = nn.Sequential(
            nn.Conv2d(in_channels=self.in_channel * 2, out_channels=self.inter_channel,
                      kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(self.inter_channel),
            nn.ReLU()
        )
        self.gx_channel = nn.Sequential(
            nn.Conv2d(in_channels=self.in_spatial, out_channels=self.inter_spatial,
                      kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(self.inter_spatial),
            nn.ReLU()
        )
        num_channel_c = 1 + self.inter_channel
        self.W_channel = nn.Sequential(
            nn.Conv2d(in_channels=num_channel_c, out_channels=num_channel_c // down_ratio,
                      kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(num_channel_c // down_ratio),
            nn.ReLU(),
            nn.Conv2d(in_channels=num_channel_c // down_ratio, out_channels=1,
                      kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(1)
        )

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.InstanceNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, x):
        temp_x = x[:, 1:].unsqueeze(-1) # b, 128, 768, 1

        theta_xc = self.theta_channel(temp_x).squeeze(-1).permute(0, 2, 1)  # b, 768, 32
        phi_xc = self.phi_channel(temp_x).squeeze(-1)   # b, 32, 768
        Gc = torch.matmul(theta_xc, phi_xc)  # b,256,256
        Gc_in = Gc.permute(0, 2, 1).unsqueeze(-1)  # b,256,256,1
        Gc_out = Gc.unsqueeze(-1)  # b,256,256,1
        Gc_joint = torch.cat((Gc_in, Gc_out), 1)  # b,512,256,1
        Gc_joint = self.gg_channel(Gc_joint)  # b,32,256,1

        g_xc = self.gx_channel(temp_x)  # b,256,256,1
        g_xc = torch.mean(g_xc, dim=1, keepdim=True)  # b,1,256,1
        yc = torch.cat((g_xc, Gc_joint), 1)  # b,33,256,1

        W_yc = self.W_channel(yc).transpose(1, 2)  # b,256,1,1
        channel_weight = torch.sigmoid(W_yc).squeeze(-1).squeeze(-1)     # b,256,64,32

        return channel_weight

class CBAM(object):
    def __init__(self, img_size=(256, 128), stride_size=(16, 16), patch_size=(16, 16), embed_dim=768, divide_length=None, **kwargs):
        self.name = 'CBAM'
        print("==========Use 'CBAM' Attention Model for Transformer!=========")
        self.img_size = img_size
        self.stride_size = stride_size
        self.embed_dim = embed_dim
        x = (img_size[0] - patch_size[0]) // stride_size[0] + 1
        y = (img_size[1] - patch_size[1]) // stride_size[1] + 1
        self.num_patch = x * y
        self.feature_size = (x, y)
        if divide_length is not None:
            self.Spatial = nn.ModuleList([])
            feature_size = ((x // divide_length), y)
            for i in range(divide_length):
                if i == (divide_length - 1):
                    feature_size = (x - (x // divide_length) * i, y)
                self.Spatial.append(SpatialAttention_CBAM(feature_size=feature_size))
        else:
            self.Spatial = SpatialAttention_CBAM(feature_size=self.feature_size)
        self.Channel = ChannelAttention_CBAM(embed_dim=embed_dim, feature_size=self.feature_size)

class ChannelAttention_CBAM(nn.Module):
    def __init__(self, embed_dim=768, feature_size=(16, 8), ratio=16):
        super(ChannelAttention_CBAM, self).__init__()
        self.feature_H = feature_size[0]
        self.feature_W = feature_size[1]
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.shared_MLP = nn.Sequential(
            nn.Conv2d(embed_dim, embed_dim // ratio, 1, bias=False),
            nn.ReLU(),
            nn.Conv2d(embed_dim // ratio, embed_dim, 1, bias=False)
        )
        # self.fc1 = nn.Conv2d(in_planes, in_planes // ratio, 1, bias=False)
        # self.relu1 = nn.ReLU()
        # self.fc2 = nn.Conv2d(in_planes // ratio, in_planes, 1, bias=False)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        temp_x = x[:, 1:]
        B, HW, C = x.shape
        temp_x = temp_x.transpose(1, 2).view(B, C, self.feature_H, -1)
        avg_out = self.shared_MLP(self.avg_pool(temp_x))# self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        max_out = self.shared_MLP(self.max_pool(temp_x))# self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = avg_out + max_out
        return self.sigmoid(out).squeeze(2).squeeze(2)

class SpatialAttention_CBAM(nn.Module):
    def __init__(self, feature_size=(16, 8), kernel_size=7):
        super(SpatialAttention_CBAM, self).__init__()
        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1
        self.feature_size = feature_size
        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        B, HW, C = x.shape
        temp_x = x.transpose(1, 2).view(B, C, self.feature_size[0], -1)
        avg_out = torch.mean(temp_x, dim=1, keepdim=True)
        max_out, _ = torch.max(temp_x, dim=1, keepdim=True)
        tx = torch.cat([avg_out, max_out], dim=1)
        tx = self.conv1(tx)
        return self.sigmoid(tx.expand_as(temp_x)).view(B, C, HW).transpose(1, 2)

def make_attention_module(name, cem=False, sem=False, **kwargs):
    attention_factory = {
        'RGA': RGA,
        'CBAM': CBAM,
        'Att_Enhance': CBAMLayer
    }
    return attention_factory[name](cem=cem, sem=sem, **kwargs)


class CBAMLayer(nn.Module):
    def __init__(self, reduction=16, spatial_kernel=7, cem=False, sem=False, **kwargs):
        super(CBAMLayer, self).__init__()
        # channel attention 压缩H,W为1
        channel = kwargs['channel']
        self.cem = cem
        self.sem = sem
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        # shared MLP
        self.mlp = nn.Sequential(
            # Conv2d比Linear方便操作
            # nn.Linear(channel, channel // reduction, bias=False)
            nn.Conv2d(channel, channel // reduction, 1, bias=False),
            # inplace=True直接替换，节省内存
            nn.ReLU(inplace=True),
            # nn.Linear(channel // reduction, channel,bias=False)
            nn.Conv2d(channel // reduction, channel, 1, bias=False)
        )
        # spatial attention
        self.conv = nn.Conv2d(2, 1, kernel_size=spatial_kernel,
                              padding=spatial_kernel // 2, bias=False)
        self.sigmoid = nn.Sigmoid()


    def forward(self, x):
        if self.sem:
            max_out = self.mlp(self.max_pool(x))
            avg_out = self.mlp(self.avg_pool(x))
            channel_out = self.sigmoid(max_out + avg_out)
            x = channel_out * x
            max_out, _ = torch.max(x, dim=1, keepdim=True)
            avg_out = torch.mean(x, dim=1, keepdim=True)
            spatial_out = self.sigmoid(self.conv(torch.cat([max_out, avg_out], dim=1)))
            x = spatial_out * x
        if self.cem:
            max_out = self.mlp(self.max_pool(x))
            avg_out = self.mlp(self.avg_pool(x))
            c_w = self.sigmoid(max_out + avg_out)
            return x, c_w
        return x

#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from dropblock import DropBlock2D
from network.BoQ import BoAQ, BoAq_pe


class BEV_Unet_Encoder(nn.Module):

    def __init__(self,n_height,dilation = 1,group_conv=False,input_batch_norm = True,circular_padding = True):
        super(BEV_Unet_Encoder, self).__init__()
        self.inc = inconv(n_height, 32, dilation, input_batch_norm, circular_padding)
        self.down1 = down(32, 64, dilation, group_conv, circular_padding)
        self.down2 = down(64, 128, dilation, group_conv, circular_padding)
        self.down3 = down(128, 256, dilation, group_conv, circular_padding)
        self.down4 = down(256, 512, dilation, group_conv, circular_padding)

        self.batchnorm = nn.BatchNorm1d(512)
        self.radial_att = nn.TransformerEncoderLayer(d_model=512, nhead=8, dim_feedforward=4*512, batch_first=True, dropout=0.)
        self.global_linear = nn.Linear(30*512, 2048)

        self.gap = nn.AdaptiveAvgPool2d((1, 1))
        
    def forward(self, x, return_lf = False,agg = None,dim = -1):
        x1 = self.inc(x) # [2, 32, 480, 360]
        x2 = self.down1(x1) # [2, 64, 240, 180]
        x3 = self.down2(x2) # [2, 128, 120, 90]
        x4 = self.down3(x3) # [2, 256, 60, 45]
        x5 = self.down4(x4) # [2, 512, 30, 22]
        
        global_descriptor = self.gap(x5).squeeze(-1).squeeze(-1) # [B, C]

        return global_descriptor


class BEV_Unet_BoAQ_Encoder(BEV_Unet_Encoder):

    def __init__(self,n_height,dilation = 1,group_conv=False,input_batch_norm = True,circular_padding = True):
        super(BEV_Unet_BoAQ_Encoder, self).__init__(n_height,dilation,group_conv,input_batch_norm,circular_padding)
        self.boaq = BoAQ(in_channels=512,proj_channels=512,num_rqueries=22,num_pqueries=30,num_layers=1,row_dim = 4)

    def forward(self, x,return_lf = False):
        x1 = self.inc(x) # [2, 32, 480, 360]
        x2 = self.down1(x1) # [2, 64, 240, 180]
        x3 = self.down2(x2) # [2, 128, 120, 90]
        x4 = self.down3(x3) # [2, 256, 60, 45]
        x5 = self.down4(x4) # [2, 512, 30, 22]

        global_descriptor,_ = self.boaq(x5)
        
        if return_lf:
            return global_descriptor,x5
        else:
            return global_descriptor  

class BEV_Unet_BoAQ_pe_Encoder(BEV_Unet_BoAQ_Encoder):
    def __init__(self, n_height, dilation=1, group_conv=False, input_batch_norm=True, circular_padding=True):
        super().__init__(n_height, dilation, group_conv, input_batch_norm, circular_padding)

        self.boaq = BoAq_pe(in_channels=512,proj_channels=512,num_rqueries=22,num_pqueries=30,num_layers=1,row_dim = 4)
    
    def forward_gap(self,x):
        x1 = self.inc(x) # [2, 32, 480, 360]
        x2 = self.down1(x1) # [2, 64, 240, 180]
        x3 = self.down2(x2) # [2, 128, 120, 90]
        x4 = self.down3(x3) # [2, 256, 60, 45]
        x5 = self.down4(x4) # [2, 512, 30, 22]

        global_descriptor = x5.flatten(-2).mean(dim = -1)

        return global_descriptor

    def forward_gmp(self,x):
        x1 = self.inc(x) # [2, 32, 480, 360]
        x2 = self.down1(x1) # [2, 64, 240, 180]
        x3 = self.down2(x2) # [2, 128, 120, 90]
        x4 = self.down3(x3) # [2, 256, 60, 45]
        x5 = self.down4(x4) # [2, 512, 30, 22]

        global_descriptor = x5.flatten(-2).max(dim = -1)[0]

        return global_descriptor
    
    def forward_am(self,x,dim = -1):
        x1 = self.inc(x) # [2, 32, 480, 360]
        x2 = self.down1(x1) # [2, 64, 240, 180]
        x3 = self.down2(x2) # [2, 128, 120, 90]
        x4 = self.down3(x3) # [2, 256, 60, 45]
        x5 = self.down4(x4) # [2, 512, 30, 22]

        global_descriptor = x5.mean(dim = dim).flatten(-2)

        return global_descriptor

    def forward(self,x,return_lf = False, agg = None,dim = None):
        if agg == "am":
            return self.forward_am(x,dim)
        elif agg == "gap":
            return self.forward_gap(x)
        elif agg == "gmp":
            return self.forward_gmp(x)
        else: # ARF in default
            return super().forward(x,return_lf)
  

class double_conv(nn.Module):
    '''(conv => BN => ReLU) * 2'''
    def __init__(self, in_ch, out_ch,group_conv,dilation=1):
        super(double_conv, self).__init__()
        if group_conv:
            self.conv = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 3, padding=1,groups = min(out_ch,in_ch)),
                nn.BatchNorm2d(out_ch),
                nn.LeakyReLU(inplace=True),
                nn.Conv2d(out_ch, out_ch, 3, padding=1,groups = out_ch),
                nn.BatchNorm2d(out_ch),
                nn.LeakyReLU(inplace=True)
            )
        else:
            self.conv = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 3, padding=1),
                nn.BatchNorm2d(out_ch),
                nn.LeakyReLU(inplace=True),
                nn.Conv2d(out_ch, out_ch, 3, padding=1),
                nn.BatchNorm2d(out_ch),
                nn.LeakyReLU(inplace=True)
            )

    def forward(self, x):
        x = self.conv(x)
        return x

class double_conv_circular(nn.Module):
    '''(conv => BN => ReLU) * 2'''
    def __init__(self, in_ch, out_ch,group_conv,dilation=1):
        super(double_conv_circular, self).__init__()
        if group_conv:
            self.conv1 = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 3, padding=(1,0),groups = min(out_ch,in_ch)),
                nn.BatchNorm2d(out_ch),
                nn.LeakyReLU(inplace=True)
            )
            self.conv2 = nn.Sequential(
                nn.Conv2d(out_ch, out_ch, 3, padding=(1,0),groups = out_ch),
                nn.BatchNorm2d(out_ch),
                nn.LeakyReLU(inplace=True)
            )
        else:
            self.conv1 = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 3, padding=(1,0)),
                nn.BatchNorm2d(out_ch),
                nn.LeakyReLU(inplace=True)
            )
            self.conv2 = nn.Sequential(
                nn.Conv2d(out_ch, out_ch, 3, padding=(1,0)),
                nn.BatchNorm2d(out_ch),
                nn.LeakyReLU(inplace=True)
            )

    def forward(self, x):
        #add circular padding
        x = F.pad(x,(1,1,0,0),mode = 'circular')
        x = self.conv1(x)
        x = F.pad(x,(1,1,0,0),mode = 'circular')
        x = self.conv2(x)
        return x


class inconv(nn.Module):
    def __init__(self, in_ch, out_ch, dilation, input_batch_norm, circular_padding):
        super(inconv, self).__init__()
        if input_batch_norm:
            if circular_padding:
                self.conv = nn.Sequential(
                    nn.BatchNorm2d(in_ch),
                    double_conv_circular(in_ch, out_ch,group_conv = False,dilation = dilation)
                )
            else:
                self.conv = nn.Sequential(
                    nn.BatchNorm2d(in_ch),
                    double_conv(in_ch, out_ch,group_conv = False,dilation = dilation)
                )
        else:
            if circular_padding:
                self.conv = double_conv_circular(in_ch, out_ch,group_conv = False,dilation = dilation)
            else:
                self.conv = double_conv(in_ch, out_ch,group_conv = False,dilation = dilation)

    def forward(self, x):
        x = self.conv(x)
        return x


class down(nn.Module):
    def __init__(self, in_ch, out_ch, dilation, group_conv, circular_padding):
        super(down, self).__init__()
        if circular_padding:
            self.mpconv = nn.Sequential(
                nn.MaxPool2d(2),
                double_conv_circular(in_ch, out_ch,group_conv = group_conv,dilation = dilation)
            )
        else:
            self.mpconv = nn.Sequential(
                nn.MaxPool2d(2),
                double_conv(in_ch, out_ch,group_conv = group_conv,dilation = dilation)
            )                

    def forward(self, x):
        x = self.mpconv(x)
        return x


class up(nn.Module):
    def __init__(self, in_ch, out_ch, circular_padding, bilinear=True, group_conv=False, use_dropblock = False, drop_p = 0.5):
        super(up, self).__init__()

        #  would be a nice idea if the upsampling could be learned too,
        #  but my machine do not have enough memory to handle all those weights
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        elif group_conv:
            self.up = nn.ConvTranspose2d(in_ch//2, in_ch//2, 2, stride=2,groups = in_ch//2)
        else:
            self.up = nn.ConvTranspose2d(in_ch//2, in_ch//2, 2, stride=2)

        if circular_padding:
            self.conv = double_conv_circular(in_ch, out_ch,group_conv = group_conv)
        else:
            self.conv = double_conv(in_ch, out_ch,group_conv = group_conv)

        self.use_dropblock = use_dropblock
        if self.use_dropblock:
            self.dropblock = DropBlock2D(block_size=7, drop_prob=drop_p)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        
        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, (diffX // 2, diffX - diffX//2,
                        diffY // 2, diffY - diffY//2))
        
        # for padding issues, see 
        # https://github.com/HaiyongJiang/U-Net-Pytorch-Unstructured-Buggy/commit/0e854509c2cea854e247a9c615f175f76fbb2e3a
        # https://github.com/xiaopeng-liao/Pytorch-UNet/commit/8ebac70e633bac59fc22bb5195e513d5832fb3bd

        x = torch.cat([x2, x1], dim=1)
        x = self.conv(x)
        if self.use_dropblock:
            x = self.dropblock(x)
        return x


class outconv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(outconv, self).__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, 1)

    def forward(self, x):
        x = self.conv(x)
        return x


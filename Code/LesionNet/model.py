import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
import torch.utils.model_zoo as model_zoo
from collections import OrderedDict

from PIL import Image
import os
import os.path
import numpy as np
import math
import ipdb

'''3D Res-Net Elements'''
def conv3x3x3(in_planes, out_planes, stride=1):
    # 3x3x3 convolution with padding
    return nn.Conv3d(
        in_planes,
        out_planes,
        kernel_size=3,
        stride=stride,
        padding=1,
        bias=False)


def downsample_basic_block(x, planes, stride):
    out = F.avg_pool3d(x, kernel_size=1, stride=stride)
    zero_pads = torch.Tensor(
        out.size(0), planes - out.size(1), out.size(2), out.size(3),
        out.size(4)).zero_()
    if isinstance(out.data, torch.cuda.FloatTensor):
        zero_pads = zero_pads.cuda()

    out =nn.Variable(torch.cat([out.data, zero_pads], dim=1))

    return out


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm3d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3x3(planes, planes)
        self.bn2 = nn.BatchNorm3d(planes)
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
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv3d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm3d(planes)
        self.conv2 = nn.Conv3d(
            planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm3d(planes)
        self.conv3 = nn.Conv3d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm3d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

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


"""
Lesion-Net: 3D U-Net (Segmentation) & 3D Res-Net (Classification)
"""
class LesionNet(nn.Module):
    def __init__(self, in_channel, n_classes, block, layers):
        super(LesionNet, self).__init__()

        '''
        3D U-Net Construction
        '''
        self.in_channel = in_channel
        self.n_classes = n_classes

        self.ec0 = self.encoder(self.in_channel, 32, bias=True, batchnorm=False)
        self.ec1 = self.encoder(32, 64, bias=True, batchnorm=False)
        self.ec2 = self.encoder(64, 64, bias=True, batchnorm=False)
        self.ec3 = self.encoder(64, 128, bias=True, batchnorm=False)
        self.ec4 = self.encoder(128, 128, bias=True, batchnorm=False)
        self.ec5 = self.encoder(128, 256, bias=True, batchnorm=False)
        self.ec6 = self.encoder(256, 256, bias=True, batchnorm=False)
        self.ec7 = self.encoder(256, 512, bias=True, batchnorm=False)

        self.pool0 = nn.MaxPool3d(2)
        self.pool1 = nn.MaxPool3d(2)
        self.pool2 = nn.MaxPool3d(2)

        self.dc9 = self.decoder(512, 512, kernel_size=2, stride=2, bias=True)

        self.dc8 = self.decoder(256 + 512, 256, kernel_size=3, stride=1, padding=1, bias=True)
        self.dc7 = self.decoder(256, 256, kernel_size=3, stride=1, padding=1, bias=True)
        self.out2 = nn.Sequential(
            nn.Conv3d(256, 1, kernel_size=3, stride=1, padding=1, bias=True),
            nn.Upsample(scale_factor=4))   # level 2 out

        self.dc6 = self.decoder(256, 256, kernel_size=2, stride=2, bias=True)

        self.dc5 = self.decoder(128 + 256, 128, kernel_size=3, stride=1, padding=1, bias=True)
        self.dc4 = self.decoder(128, 128, kernel_size=3, stride=1, padding=1, bias=True)
        self.out1 = nn.Sequential(
            nn.Conv3d(128, 1, kernel_size=3, stride=1, padding=1, bias=True),
            nn.Upsample(scale_factor=2))   # level 1 out

        self.dc3 = self.decoder(128, 128, kernel_size=2, stride=2, bias=True)

        self.dc2 = self.decoder(64 + 128, 64, kernel_size=3, stride=1, padding=1, bias=True)
        self.dc1 = self.decoder(64, 64, kernel_size=3, stride=1, padding=1, bias=True)
        self.out0 = self.decoder(64, 1, kernel_size=1, stride=1, bias=True)     # level 0 out

        '''
        3D Res-Net Construction
        '''
        self.conv_brig = nn.Conv3d(1, 3, kernel_size=3, stride=(1, 1, 1), padding=(1, 1, 1), bias=True)

        self.inplanes = 64
        self.conv1 = nn.Conv3d(3, 64, kernel_size=7, stride=(1, 2, 2), padding=(3, 3, 3), bias=True)
        self.bn1 = nn.BatchNorm3d(64)
        self.relu = nn.ReLU(inplace=True)

        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 64, layers[1], stride=1)
        self.layer3 = self._make_layer(block, 64, layers[2], stride=1)

        self.layer_hm = nn.Conv3d(256, 8, kernel_size=3, stride=(1, 1, 1), padding=(1, 1, 1), bias=True)

        

    '''3D U-Net Elements'''
    def encoder(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=True, batchnorm=False):
        if batchnorm:
            layer = nn.Sequential(
                nn.Conv3d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, bias=bias),
                nn.BatchNorm3d(out_channels),
                nn.ReLU())
        else:
            layer = nn.Sequential(
                nn.Conv3d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, bias=bias),
                nn.ReLU())
        return layer


    def decoder(self, in_channels, out_channels, kernel_size, stride=1, padding=0, output_padding=0, bias=True):
        layer = nn.Sequential(
            nn.ConvTranspose3d(in_channels, out_channels, kernel_size, stride=stride,
                               padding=padding, output_padding=output_padding, bias=bias),
            nn.ReLU())
        return layer

    '''3D Res-Net Elements'''
    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv3d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False), nn.BatchNorm3d(planes * block.expansion))

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        '''Tumor Attention Net (3D U-Net)'''
        e0 = self.ec0(x)
        syn0 = self.ec1(e0)

        e1 = self.pool0(syn0)
        e2 = self.ec2(e1)
        syn1 = self.ec3(e2)

        e3 = self.pool1(syn1)
        e4 = self.ec4(e3)
        syn2 = self.ec5(e4)

        e5 = self.pool2(syn2)
        e6 = self.ec6(e5)
        e7 = self.ec7(e6)

        d9 = torch.cat((self.dc9(e7), syn2), dim=1)

        d8 = self.dc8(d9)
        d7 = self.dc7(d8)
        out2 = self.out2(d7)

        d6 = torch.cat((self.dc6(d7), syn1), dim=1)

        d5 = self.dc5(d6)
        d4 = self.dc4(d5)
        out1 = self.out1(d4)

        d3 = torch.cat((self.dc3(d4), syn0), dim=1)

        d2 = self.dc2(d3)
        d1 = self.dc1(d2)

        out0 = self.out0(d1)  # Tumor Attention Output - Supervised by 3D Binary Mask

        '''Tumor Classification Network (3D Res-Net50)'''
        x = self.conv_brig(out0)  # 1 channel to 3 channels

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)

        hm = self.layer_hm(x)  # heatmap/attention for classification output

        return out0, out1, out2, hm


def lesion_net(pretrained=False, **kwargs):
    """
    Args:
        fix_para (bool): If True, Fix the weights in part of the CNN
    """
    # model = LesionNet(**kwargs)
    model = LesionNet(in_channel=1, n_classes=8, block=Bottleneck, layers=[3, 4, 6])

    # Optional: Fix weights in certain layers
    if pretrained is False:
        for para in model.parameters():
            para.requires_grad = True

    return model

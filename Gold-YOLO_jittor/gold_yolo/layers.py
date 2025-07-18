# Jittor implementation of Gold-YOLO specific layers
# Migrated from PyTorch version

import jittor as jt
from jittor import nn
import numpy as np


class Conv(nn.Module):
    """Normal Conv with SiLU activation"""
    
    def __init__(self, in_channels, out_channels, kernel_size, stride, groups=1, bias=False, padding=None):
        super().__init__()
        
        if padding is None:
            padding = kernel_size // 2
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            groups=groups,
            bias=bias,
        )
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = nn.SiLU()
    
    def execute(self, x):
        return self.act(self.bn(self.conv(x)))


class Conv2d_BN(nn.Sequential):
    """Conv2d with BatchNorm"""
    
    def __init__(self, a, b, ks=1, stride=1, pad=0, dilation=1,
                 groups=1, bn_weight_init=1):
        super().__init__()
        self.inp_channel = a
        self.out_channel = b
        self.ks = ks
        self.pad = pad
        self.stride = stride
        self.dilation = dilation
        self.groups = groups
        
        self.add_module('c', nn.Conv2d(
            a, b, ks, stride, pad, dilation, groups, bias=False))
        bn = nn.BatchNorm2d(b)
        jt.init.constant_(bn.weight, bn_weight_init)
        jt.init.constant_(bn.bias, 0)
        self.add_module('bn', bn)


def drop_path(x, drop_prob: float = 0., training: bool = False):
    """Drop paths (Stochastic Depth) per sample"""
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)
    random_tensor = keep_prob + jt.rand(shape, dtype=x.dtype)
    random_tensor = jt.floor(random_tensor)  # binarize
    output = x / keep_prob * random_tensor
    return output


class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample"""
    
    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob
    
    def execute(self, x):
        return drop_path(x, self.drop_prob, self.training)


class h_sigmoid(nn.Module):
    """Hard sigmoid activation"""
    
    def __init__(self, inplace=True):
        super(h_sigmoid, self).__init__()
        self.relu = nn.ReLU6()
    
    def execute(self, x):
        return self.relu(x + 3) / 6

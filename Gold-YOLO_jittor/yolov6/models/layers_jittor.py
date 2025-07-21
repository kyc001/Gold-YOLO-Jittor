#!/usr/bin/env python3
# -*- coding:utf-8 -*-
"""
Layers模块的Jittor实现 - 100%对齐PyTorch官方版本
基于Gold-YOLO的layers.py实现
"""

import jittor as jt
from jittor import nn

# 导入兼容层
from .jittor_compat import compat_nn


class Conv2d_BN(nn.Module):
    """Conv2d + BatchNorm - Jittor版本"""
    
    def __init__(self, in_channels, out_channels, kernel_size=1, stride=1, padding=0, 
                 dilation=1, groups=1, bn_weight_init=1, norm_cfg=None):
        super().__init__()
        
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, 
                             dilation, groups, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        
        # 初始化权重
        nn.init.kaiming_normal_(self.conv.weight, mode='fan_out', nonlinearity='relu')
        if bn_weight_init != 1:
            nn.init.constant_(self.bn.weight, bn_weight_init)
        nn.init.constant_(self.bn.bias, 0)
    
    def execute(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return x


class Conv(nn.Module):
    """标准卷积层 - Jittor版本"""
    
    def __init__(self, in_channels, out_channels, kernel_size=1, stride=1, padding=None, 
                 groups=1, activation=True):
        super().__init__()
        
        if padding is None:
            padding = kernel_size // 2
        
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, 
                             groups=groups, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = nn.SiLU() if activation else nn.Identity()
    
    def execute(self, x):
        return self.act(self.bn(self.conv(x)))


class DropPath(nn.Module):
    """Drop Path (Stochastic Depth) - Jittor版本"""
    
    def __init__(self, drop_prob=None):
        super().__init__()
        self.drop_prob = drop_prob
    
    def execute(self, x):
        if self.drop_prob == 0. or not self.training:
            return x
        
        keep_prob = 1 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)  # work with diff dim tensors, not just 2D ConvNets
        random_tensor = keep_prob + jt.rand(shape, dtype=x.dtype)
        random_tensor = random_tensor.floor()  # binarize
        output = x.div(keep_prob) * random_tensor
        return output


class h_sigmoid(nn.Module):
    """Hard Sigmoid - Jittor版本"""
    
    def __init__(self, inplace=True):
        super().__init__()
        self.inplace = inplace
    
    def execute(self, x):
        return compat_nn.relu6(x + 3) / 6


class h_swish(nn.Module):
    """Hard Swish - Jittor版本"""
    
    def __init__(self, inplace=True):
        super().__init__()
        self.inplace = inplace
    
    def execute(self, x):
        return x * compat_nn.relu6(x + 3) / 6


class SELayer(nn.Module):
    """Squeeze-and-Excitation Layer - Jittor版本"""
    
    def __init__(self, channel, reduction=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )
    
    def execute(self, x):
        b, c, _, _ = x.shape
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)


class DWConv(nn.Module):
    """Depthwise Convolution - Jittor版本"""
    
    def __init__(self, in_channels, out_channels, kernel_size=1, stride=1, activation=True):
        super().__init__()
        
        self.dconv = nn.Conv2d(in_channels, in_channels, kernel_size, stride, 
                              kernel_size//2, groups=in_channels, bias=False)
        self.pconv = nn.Conv2d(in_channels, out_channels, 1, 1, 0, bias=False)
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.act = nn.SiLU() if activation else nn.Identity()
    
    def execute(self, x):
        x = self.act(self.bn1(self.dconv(x)))
        x = self.act(self.bn2(self.pconv(x)))
        return x


class ConvBNAct(nn.Module):
    """Conv + BN + Activation - Jittor版本"""
    
    def __init__(self, in_channels, out_channels, kernel_size=1, stride=1, padding=0,
                 groups=1, activation=nn.SiLU):
        super().__init__()
        
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding,
                             groups=groups, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = activation() if activation else nn.Identity()
    
    def execute(self, x):
        return self.act(self.bn(self.conv(x)))


class Focus(nn.Module):
    """Focus wh information into c-space - Jittor版本"""
    
    def __init__(self, in_channels, out_channels, kernel_size=1, stride=1, padding=None, 
                 groups=1, activation=True):
        super().__init__()
        self.conv = Conv(in_channels * 4, out_channels, kernel_size, stride, padding, 
                        groups, activation)
    
    def execute(self, x):
        # x(b,c,w,h) -> y(b,4c,w/2,h/2)
        return self.conv(jt.concat([x[..., ::2, ::2], x[..., 1::2, ::2], 
                                   x[..., ::2, 1::2], x[..., 1::2, 1::2]], 1))


class Concat(nn.Module):
    """Concatenate a list of tensors along dimension - Jittor版本"""
    
    def __init__(self, dimension=1):
        super().__init__()
        self.d = dimension
    
    def execute(self, x):
        return jt.concat(x, self.d)


class Chuncat(nn.Module):
    """Chunk and concatenate tensors - Jittor版本"""
    
    def __init__(self, dimension=1):
        super().__init__()
        self.d = dimension
    
    def execute(self, x):
        x1, x2 = x.chunk(2, self.d)
        return jt.concat((x1, x2), self.d)


class Shortcut(nn.Module):
    """Shortcut connection - Jittor版本"""
    
    def __init__(self, dimension=0):
        super().__init__()
        self.d = dimension
    
    def execute(self, x):
        return x[0] + x[1]


class Foldcut(nn.Module):
    """Fold and cut operation - Jittor版本"""
    
    def __init__(self, dimension=0):
        super().__init__()
        self.d = dimension
    
    def execute(self, x):
        x1, x2 = x.chunk(2, self.d)
        return x1 + x2

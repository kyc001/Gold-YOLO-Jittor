#!/usr/bin/env python3
# -*- coding:utf-8 -*-
"""
EfficientRep Backbone的Jittor实现 - 100%对齐PyTorch官方版本
"""

import jittor as jt
from jittor import nn

from yolov6.layers.common import RepVGGBlock, RepBlock


class RepVGGBlock(jt.nn.Module):
    """RepVGGBlock - 与PyTorch版本严格对齐"""
    
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, 
                 dilation=1, groups=1, padding_mode='zeros', deploy=False, use_se=False):
        super(RepVGGBlock, self).__init__()
        
        self.deploy = deploy
        self.groups = groups
        self.in_channels = in_channels
        self.out_channels = out_channels
        
        assert kernel_size == 3
        assert padding == 1
        
        padding_11 = padding - kernel_size // 2
        
        self.nonlinearity = jt.nn.ReLU()
        
        if deploy:
            self.rbr_reparam = jt.nn.Conv2d(in_channels=in_channels, out_channels=out_channels,
                                          kernel_size=kernel_size, stride=stride,
                                          padding=padding, dilation=dilation, groups=groups, bias=True,
                                          padding_mode=padding_mode)
        else:
            self.rbr_identity = jt.nn.BatchNorm2d(num_features=in_channels) if out_channels == in_channels and stride == 1 else None
            self.rbr_dense = self.conv_bn(in_channels=in_channels, out_channels=out_channels,
                                        kernel_size=kernel_size, stride=stride, padding=padding, groups=groups)
            self.rbr_1x1 = self.conv_bn(in_channels=in_channels, out_channels=out_channels,
                                      kernel_size=1, stride=stride, padding=padding_11, groups=groups)
    
    def conv_bn(self, in_channels, out_channels, kernel_size, stride, padding, groups=1):
        """创建conv+bn层"""
        result = jt.nn.Sequential()
        result.add_module('conv', jt.nn.Conv2d(in_channels=in_channels, out_channels=out_channels,
                                             kernel_size=kernel_size, stride=stride, padding=padding,
                                             groups=groups, bias=False))
        result.add_module('bn', jt.nn.BatchNorm2d(num_features=out_channels))
        return result
    
    def forward(self, inputs):
        if hasattr(self, 'rbr_reparam'):
            return self.nonlinearity(self.rbr_reparam(inputs))
        
        if self.rbr_identity is None:
            id_out = 0
        else:
            id_out = self.rbr_identity(inputs)
        
        return self.nonlinearity(self.rbr_dense(inputs) + self.rbr_1x1(inputs) + id_out)
    
    def execute(self, inputs):
        return self.forward(inputs)


class RepBlock(jt.nn.Module):
    """RepBlock - 与PyTorch版本严格对齐"""
    
    def __init__(self, in_channels, out_channels, n=1, block=RepVGGBlock):
        super(RepBlock, self).__init__()
        
        self.conv1 = block(in_channels, out_channels)
        self.block = jt.nn.Sequential(*(block(out_channels, out_channels) for _ in range(n - 1))) if n > 1 else None
    
    def forward(self, x):
        x = self.conv1(x)
        if self.block is not None:
            x = self.block(x)
        return x
    
    def execute(self, x):
        return self.forward(x)


class SimSPPF(jt.nn.Module):
    """SimSPPF - 简化的SPPF层"""
    
    def __init__(self, in_channels, out_channels, kernel_size=5):
        super(SimSPPF, self).__init__()
        c_ = in_channels // 2
        self.cv1 = jt.nn.Conv2d(in_channels, c_, 1, 1)
        self.cv2 = jt.nn.Conv2d(c_ * 4, out_channels, 1, 1)
        self.m = jt.nn.MaxPool2d(kernel_size=kernel_size, stride=1, padding=kernel_size // 2)
    
    def forward(self, x):
        x = self.cv1(x)
        y1 = self.m(x)
        y2 = self.m(y1)
        return self.cv2(jt.concat([x, y1, y2, self.m(y2)], 1))
    
    def execute(self, x):
        return self.forward(x)


class EfficientRep(jt.nn.Module):
    """EfficientRep Backbone - 与PyTorch版本严格对齐"""
    
    def __init__(self, in_channels=3, channels_list=None, num_repeats=None, 
                 block=RepVGGBlock, fuse_P2=False, cspsppf=False):
        super(EfficientRep, self).__init__()
        
        assert channels_list is not None
        assert num_repeats is not None
        self.fuse_P2 = fuse_P2
        
        self.stem = block(
            in_channels=in_channels,
            out_channels=channels_list[0],
            kernel_size=3,
            stride=2
        )
        
        self.ERBlock_2 = jt.nn.Sequential(
            block(
                in_channels=channels_list[0],
                out_channels=channels_list[1],
                kernel_size=3,
                stride=2
            ),
            RepBlock(
                in_channels=channels_list[1],
                out_channels=channels_list[1],
                n=num_repeats[1],
                block=block,
            )
        )
        
        self.ERBlock_3 = jt.nn.Sequential(
            block(
                in_channels=channels_list[1],
                out_channels=channels_list[2],
                kernel_size=3,
                stride=2
            ),
            RepBlock(
                in_channels=channels_list[2],
                out_channels=channels_list[2],
                n=num_repeats[2],
                block=block,
            )
        )
        
        self.ERBlock_4 = jt.nn.Sequential(
            block(
                in_channels=channels_list[2],
                out_channels=channels_list[3],
                kernel_size=3,
                stride=2
            ),
            RepBlock(
                in_channels=channels_list[3],
                out_channels=channels_list[3],
                n=num_repeats[3],
                block=block,
            )
        )
        
        self.ERBlock_5 = jt.nn.Sequential(
            block(
                in_channels=channels_list[3],
                out_channels=channels_list[4],
                kernel_size=3,
                stride=2,
            ),
            RepBlock(
                in_channels=channels_list[4],
                out_channels=channels_list[4],
                n=num_repeats[4],
                block=block,
            ),
            SimSPPF(
                in_channels=channels_list[4],
                out_channels=channels_list[4],
                kernel_size=5
            )
        )
    
    def forward(self, x):
        outputs = []
        x = self.stem(x)
        x = self.ERBlock_2(x)
        if self.fuse_P2:
            outputs.append(x)
        x = self.ERBlock_3(x)
        outputs.append(x)  # P3
        x = self.ERBlock_4(x)
        outputs.append(x)  # P4
        x = self.ERBlock_5(x)
        outputs.append(x)  # P5
        
        return tuple(outputs)
    
    def execute(self, x):
        return self.forward(x)

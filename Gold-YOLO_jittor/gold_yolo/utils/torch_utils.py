#!/usr/bin/env python3
# -*- coding:utf-8 -*-
"""
PyTorch工具函数的Jittor实现
"""

import jittor as jt
import jittor.nn as nn

def initialize_weights(model):
    """初始化模型权重"""
    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, 0, 0.01)
            nn.init.constant_(m.bias, 0)

def get_block(training_mode):
    """获取训练模式对应的block"""
    from ..layers.common import RepVGGBlock, Conv
    
    if training_mode == "repvgg":
        return RepVGGBlock
    else:
        return Conv

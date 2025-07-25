#!/usr/bin/env python3
# -*- coding:utf-8 -*-
"""
GOLD-YOLO Jittor版本 - RepVGG优化器
从PyTorch版本迁移到Jittor框架，简化实现
"""

import numpy as np
import jittor as jt
import jittor.nn as nn
from yolov6.utils.events import LOGGER


def extract_blocks_into_list(model, blocks):
    """提取模型中的RepVGG块"""
    for module in model.children():
        if hasattr(module, 'scale_identity') or hasattr(module, 'scale_1x1'):
            blocks.append(module)
        else:
            extract_blocks_into_list(module, blocks)


def extract_scales(model):
    """提取RepVGG块的缩放参数"""
    blocks = []
    if isinstance(model, dict) and 'model' in model:
        extract_blocks_into_list(model['model'], blocks)
    else:
        extract_blocks_into_list(model, blocks)
    
    scales = []
    for b in blocks:
        if hasattr(b, 'scale_identity'):
            scales.append((
                b.scale_identity.weight.detach() if hasattr(b.scale_identity, 'weight') else None,
                b.scale_1x1.weight.detach() if hasattr(b.scale_1x1, 'weight') else None,
                b.scale_conv.weight.detach() if hasattr(b.scale_conv, 'weight') else None
            ))
        elif hasattr(b, 'scale_1x1'):
            scales.append((
                b.scale_1x1.weight.detach() if hasattr(b.scale_1x1, 'weight') else None,
                b.scale_conv.weight.detach() if hasattr(b.scale_conv, 'weight') else None
            ))
        
        if scales and len(scales[-1]) >= 2:
            valid_scales = [s for s in scales[-1] if s is not None]
            if len(valid_scales) >= 2:
                LOGGER.info(f'extract scales: {valid_scales[-2].mean():.6f}, {valid_scales[-1].mean():.6f}')
    
    return scales


def check_keywords_in_name(name, keywords=()):
    """检查名称中是否包含关键词"""
    isin = False
    for keyword in keywords:
        if keyword in name:
            isin = True
    return isin


def set_weight_decay(model, skip_list=(), skip_keywords=(), echo=False):
    """设置权重衰减参数"""
    has_decay = []
    no_decay = []
    
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue  # frozen weights
        if 'identity.weight' in name:
            has_decay.append(param)
            if echo:
                LOGGER.info(f"{name} USE weight decay")
        elif len(param.shape) == 1 or name.endswith(".bias") or name in skip_list or \
                check_keywords_in_name(name, skip_keywords):
            no_decay.append(param)
            if echo:
                LOGGER.info(f"{name} NO weight decay")
        else:
            has_decay.append(param)
            if echo:
                LOGGER.info(f"{name} USE weight decay")
    
    return [{'params': has_decay}, {'params': no_decay, 'weight_decay': 0.}]


class RepVGGOptimizer:
    """RepVGG优化器包装器"""
    
    def __init__(self, optimizer):
        self.optimizer = optimizer
    
    def step(self):
        """执行优化步骤"""
        return self.optimizer.step()
    
    def zero_grad(self):
        """清零梯度"""
        return self.optimizer.zero_grad()
    
    def state_dict(self):
        """获取状态字典"""
        return self.optimizer.state_dict()
    
    def load_state_dict(self, state_dict):
        """加载状态字典"""
        return self.optimizer.load_state_dict(state_dict)
    
    @property
    def param_groups(self):
        """获取参数组"""
        return self.optimizer.param_groups

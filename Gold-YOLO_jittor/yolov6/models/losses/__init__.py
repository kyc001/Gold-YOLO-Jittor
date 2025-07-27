#!/usr/bin/env python3
# -*- coding:utf-8 -*-
"""
GOLD-YOLO Jittor版本 - 损失函数包
严格对齐PyTorch版本，百分百还原所有功能
"""

# 导入主要的损失函数类
# 临时解决方案：直接从上级目录导入修复版本
import sys
import os
parent_dir = os.path.dirname(os.path.dirname(__file__))
losses_file = os.path.join(parent_dir, 'losses.py')

# 动态导入修复版本的ComputeLoss
import importlib.util
spec = importlib.util.spec_from_file_location("fixed_losses", losses_file)
fixed_losses = importlib.util.module_from_spec(spec)
spec.loader.exec_module(fixed_losses)
ComputeLoss = fixed_losses.ComputeLoss

# 导入其他损失函数
from .loss_fuseab import ComputeLoss as ComputeLoss_ab
from .loss_distill import ComputeLoss as ComputeLoss_distill
from .loss_distill_ns import ComputeLoss as ComputeLoss_distill_ns

# 导出所有损失函数
__all__ = [
    'ComputeLoss',
    'ComputeLoss_ab',
    'ComputeLoss_distill',
    'ComputeLoss_distill_ns',
    'create_loss_function'
]


def create_loss_function(cfg, num_classes=20):
    """创建损失函数"""
    return ComputeLoss(
        fpn_strides=[8, 16, 32],
        grid_cell_size=5.0,
        grid_cell_offset=0.5,
        num_classes=num_classes,
        ori_img_size=640,
        warmup_epoch=4,
        use_dfl=cfg.get('use_dfl', False),
        reg_max=cfg.get('reg_max', 0),
        iou_type='giou',
        loss_weight={
            'class': 1.0,
            'iou': 2.5,
            'dfl': 0.5
        }
    )

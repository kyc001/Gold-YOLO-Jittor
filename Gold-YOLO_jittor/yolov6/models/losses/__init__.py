#!/usr/bin/env python3
# -*- coding:utf-8 -*-
"""
GOLD-YOLO Jittor版本 - 损失函数包
严格对齐PyTorch版本，百分百还原所有功能
"""

# 导入主要的损失函数类
from .loss import ComputeLoss
from .loss_fuseab import ComputeLoss as ComputeLoss_ab
from .loss_distill import ComputeLoss as ComputeLoss_distill
from .loss_distill_ns import ComputeLoss as ComputeLoss_distill_ns

# 导出所有损失函数
__all__ = [
    'ComputeLoss',
    'ComputeLoss_ab',
    'ComputeLoss_distill',
    'ComputeLoss_distill_ns'
]

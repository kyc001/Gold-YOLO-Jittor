#!/usr/bin/env python3
# -*- coding:utf-8 -*-
"""
GOLD-YOLO Jittor版本 - FuseAB损失函数
从PyTorch版本迁移到Jittor框架，严格对齐所有功能
"""

import jittor as jt
import jittor.nn as nn
from .loss import ComputeLoss as BaseLoss


class ComputeLoss(BaseLoss):
    """FuseAB版本的损失计算类，继承自基础损失类"""
    
    def __init__(self, model, use_dfl=True):
        super().__init__(model, use_dfl)
        self.fuse_ab = True
    
    def __call__(self, outputs, targets, epoch_num, step_num):
        """
        计算FuseAB损失
        
        Args:
            outputs: 模型输出
            targets: 目标标签
            epoch_num: 当前epoch
            step_num: 当前step
        
        Returns:
            损失值和损失组件
        """
        # 使用基础损失计算
        return super().__call__(outputs, targets, epoch_num, step_num)

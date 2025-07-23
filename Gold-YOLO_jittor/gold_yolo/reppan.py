#!/usr/bin/env python3
# -*- coding:utf-8 -*-
"""
RepPAN - 与PyTorch版本对齐的RepGD Neck
新芽第二阶段：项目结构整理，与PyTorch版本完全对齐
"""

# 导入增强版RepGD Neck作为RepPAN的实现
from .models.enhanced_repgd_neck import EnhancedRepGDNeck

# 为了与PyTorch版本对齐，创建RepPAN别名
RepPAN = EnhancedRepGDNeck
RepPANNeck = EnhancedRepGDNeck

# 导出函数
def build_reppan_neck(channels_list, num_repeats, block, extra_cfg=None):
    """构建RepPAN Neck - 与PyTorch版本对齐的接口"""
    return EnhancedRepGDNeck(
        channels_list=channels_list,
        num_repeats=num_repeats,
        block=block,
        extra_cfg=extra_cfg
    )

# 导出所有组件
__all__ = ['RepPAN', 'RepPANNeck', 'EnhancedRepGDNeck', 'build_reppan_neck']

print("✅ Gold-YOLO RepPAN模块加载完成 - 与PyTorch版本对齐")

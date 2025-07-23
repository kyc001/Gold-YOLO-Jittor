#!/usr/bin/env python3
# -*- coding:utf-8 -*-
"""
Gold-YOLO Common - 与PyTorch版本对齐的通用组件
新芽第二阶段：项目结构整理，与PyTorch版本完全对齐
"""

# 导入所有通用组件
from .layers.common import *

# 为了与PyTorch版本对齐，重新导出所有组件
__all__ = [
    # 基础卷积组件
    'Conv', 'RepVGGBlock', 'RepBlock', 'SimConv', 'SimSPPF', 'CSPSPPF',
    
    # 转置和融合组件
    'Transpose', 'SimFusion_3in', 'SimFusion_4in', 'AdvPoolFusion',
    
    # 高级组件
    'InjectionMultiSum_Auto_pool', 'BepC3', 'BottleRep', 'ConvWrapper',
    
    # 激活函数
    'SiLU', 'Hardswish', 'MemoryEfficientSwish',
    
    # 工具函数
    'make_divisible', 'autopad'
]

print("✅ Gold-YOLO common模块加载完成 - 与PyTorch版本对齐")

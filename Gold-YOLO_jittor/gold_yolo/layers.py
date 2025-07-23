#!/usr/bin/env python3
# -*- coding:utf-8 -*-
"""
Gold-YOLO Layers - 与PyTorch版本对齐
新芽第二阶段：项目结构整理，与PyTorch版本完全对齐
"""

# 导入所有layers模块的内容
from .layers.common import *
from .layers.transformer import *
from .layers.advanced_fusion import *

# 为了与PyTorch版本对齐，导出所有常用组件
__all__ = [
    # 基础组件
    'Conv', 'RepVGGBlock', 'RepBlock', 'SimConv', 'SimSPPF', 'CSPSPPF',
    'Transpose', 'SimFusion_3in', 'SimFusion_4in', 'AdvPoolFusion',
    'InjectionMultiSum_Auto_pool', 'BepC3', 'BottleRep', 'ConvWrapper',
    
    # Transformer组件
    'PyramidPoolAgg', 'TopBasicLayer', 'C2T_Attention',
    
    # 高级融合组件
    'AdvancedFusion', 'MultiScaleFusion'
]

print("✅ Gold-YOLO layers模块加载完成 - 与PyTorch版本对齐")

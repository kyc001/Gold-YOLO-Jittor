#!/usr/bin/env python3
# -*- coding:utf-8 -*-
"""
Gold-YOLO 层模块初始化
新芽第二阶段：完整架构实现
"""

from .common import (
    Conv, RepVGGBlock, RepBlock, SimConv, SimSPPF, CSPSPPF,
    Transpose, SimFusion_3in, SimFusion_4in, AdvPoolFusion,
    InjectionMultiSum_Auto_pool, BepC3, BottleRep, ConvWrapper
)

from .transformer import (
    PyramidPoolAgg, TopBasicLayer, C2T_Attention,
    MultiHeadAttention, MLP, TransformerBlock
)

__all__ = [
    # Common layers
    'Conv', 'RepVGGBlock', 'RepBlock', 'SimConv', 'SimSPPF', 'CSPSPPF',
    'Transpose', 'SimFusion_3in', 'SimFusion_4in', 'AdvPoolFusion',
    'InjectionMultiSum_Auto_pool', 'BepC3', 'BottleRep', 'ConvWrapper',
    
    # Transformer layers
    'PyramidPoolAgg', 'TopBasicLayer', 'C2T_Attention',
    'MultiHeadAttention', 'MLP', 'TransformerBlock'
]

#!/usr/bin/env python3
# -*- coding:utf-8 -*-
"""
Gold-YOLO Transformer - 与PyTorch版本对齐的Transformer组件
新芽第二阶段：项目结构整理，与PyTorch版本完全对齐
"""

# 导入所有transformer组件
from .layers.transformer import *

# 为了与PyTorch版本对齐，重新导出所有组件
__all__ = [
    # 核心Transformer组件
    'PyramidPoolAgg', 'TopBasicLayer', 'C2T_Attention',
    
    # 注意力机制
    'MultiHeadAttention', 'SelfAttention', 'CrossAttention',
    
    # 位置编码
    'PositionalEncoding', 'LearnablePositionalEncoding',
    
    # 工具函数
    'drop_path', 'trunc_normal_'
]

print("✅ Gold-YOLO transformer模块加载完成 - 与PyTorch版本对齐")

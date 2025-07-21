#!/usr/bin/env python3
# -*- coding:utf-8 -*-
"""
Jittor兼容层 - 提供PyTorch风格的API
解决Jittor与PyTorch API差异问题
"""

import jittor as jt
from jittor import nn
import numpy as np


def adaptive_avg_pool2d(input, output_size):
    """自适应平均池化 - Jittor兼容版本"""
    if isinstance(output_size, int):
        output_size = (output_size, output_size)
    
    # 获取输入尺寸
    N, C, H, W = input.shape
    target_h, target_w = output_size
    
    # 计算池化核大小和步长
    kernel_h = H // target_h
    kernel_w = W // target_w
    stride_h = kernel_h
    stride_w = kernel_w
    
    # 使用平均池化
    if kernel_h > 0 and kernel_w > 0:
        return jt.nn.avg_pool2d(input, kernel_size=(kernel_h, kernel_w), stride=(stride_h, stride_w))
    else:
        return input


def interpolate(input, size=None, scale_factor=None, mode='bilinear', align_corners=False):
    """插值函数 - Jittor兼容版本"""
    if size is not None:
        if isinstance(size, int):
            size = (size, size)
        target_h, target_w = size
    elif scale_factor is not None:
        _, _, H, W = input.shape
        target_h = int(H * scale_factor)
        target_w = int(W * scale_factor)
    else:
        raise ValueError("Either size or scale_factor must be specified")
    
    # 使用Jittor的resize函数
    return jt.nn.interpolate(input, size=(target_h, target_w), mode=mode)


def softmax(input, dim=-1):
    """Softmax函数 - Jittor兼容版本"""
    return jt.nn.softmax(input, dim=dim)


def relu6(input):
    """ReLU6函数 - Jittor兼容版本"""
    return jt.clamp(jt.nn.relu(input), max_v=6.0)


def avg_pool2d(input, kernel_size, stride=None, padding=0):
    """平均池化 - Jittor兼容版本"""
    if stride is None:
        stride = kernel_size
    return jt.nn.avg_pool2d(input, kernel_size=kernel_size, stride=stride, padding=padding)


# 创建兼容的nn模块
class CompatNN:
    """兼容的nn模块"""
    
    @staticmethod
    def adaptive_avg_pool2d(input, output_size):
        return adaptive_avg_pool2d(input, output_size)
    
    @staticmethod
    def interpolate(input, size=None, scale_factor=None, mode='bilinear', align_corners=False):
        return interpolate(input, size, scale_factor, mode, align_corners)
    
    @staticmethod
    def softmax(input, dim=-1):
        return softmax(input, dim)
    
    @staticmethod
    def relu6(input):
        return relu6(input)
    
    @staticmethod
    def avg_pool2d(input, kernel_size, stride=None, padding=0):
        return avg_pool2d(input, kernel_size, stride, padding)


# 全局兼容nn对象
compat_nn = CompatNN()


def patch_jittor_nn():
    """为jittor.nn添加兼容函数"""
    if not hasattr(jt.nn, 'adaptive_avg_pool2d'):
        jt.nn.adaptive_avg_pool2d = adaptive_avg_pool2d
    
    if not hasattr(jt.nn, 'interpolate'):
        jt.nn.interpolate = interpolate
    
    if not hasattr(jt.nn, 'softmax'):
        jt.nn.softmax = softmax
    
    if not hasattr(jt.nn, 'relu6'):
        jt.nn.relu6 = relu6
    
    if not hasattr(jt.nn, 'avg_pool2d'):
        jt.nn.avg_pool2d = avg_pool2d


# 自动应用补丁
patch_jittor_nn()


# 额外的工具函数
def get_shape(tensor):
    """获取张量形状 - 兼容版本"""
    return tensor.shape


def matmul(a, b):
    """矩阵乘法 - 兼容版本"""
    return jt.matmul(a, b)


def concat(tensors, dim=0):
    """张量拼接 - 兼容版本"""
    return jt.concat(tensors, dim=dim)


def split(tensor, split_size_or_sections, dim=0):
    """张量分割 - 兼容版本"""
    if isinstance(split_size_or_sections, int):
        # 等分
        total_size = tensor.shape[dim]
        num_splits = total_size // split_size_or_sections
        sections = [split_size_or_sections] * num_splits
        if total_size % split_size_or_sections != 0:
            sections.append(total_size % split_size_or_sections)
    else:
        sections = split_size_or_sections
    
    # 使用累积索引进行分割
    splits = []
    start = 0
    for size in sections:
        if dim == 0:
            splits.append(tensor[start:start+size])
        elif dim == 1:
            splits.append(tensor[:, start:start+size])
        elif dim == 2:
            splits.append(tensor[:, :, start:start+size])
        elif dim == 3:
            splits.append(tensor[:, :, :, start:start+size])
        else:
            raise NotImplementedError(f"Split along dim {dim} not implemented")
        start += size
    
    return splits


def linspace(start, end, steps):
    """线性空间 - 兼容版本"""
    return jt.array(np.linspace(start, end, steps))


def zeros(shape):
    """零张量 - 兼容版本"""
    return jt.zeros(shape)


def ones(shape):
    """一张量 - 兼容版本"""
    return jt.ones(shape)


def randn(*shape):
    """随机正态分布张量 - 兼容版本"""
    return jt.randn(shape)


def rand(*shape):
    """随机均匀分布张量 - 兼容版本"""
    return jt.rand(shape)


# 为jt添加兼容函数
if not hasattr(jt, 'split'):
    jt.split = split

if not hasattr(jt, 'linspace'):
    jt.linspace = linspace

print("✅ Jittor兼容层加载完成")

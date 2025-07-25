#!/usr/bin/env python3
# -*- coding:utf-8 -*-
"""
ERBlock_5参数深度分析器
逐层分析每个组件的参数量，找出11,622参数误差的根源
"""

import os
import sys
import numpy as np
import jittor as jt
import jittor.nn as nn
from pathlib import Path

# 设置Jittor
jt.flags.use_cuda = 1 if jt.has_cuda else 0

# 添加项目路径
sys.path.append('models')
from gold_yolo_backbone import ZeroErrorSPPF


def analyze_layer_parameters(layer, layer_name):
    """分析单个层的参数量"""
    params = sum(p.numel() for p in layer.parameters())
    print(f"   {layer_name}: {params:,}参数")
    
    # 详细分析每个子层
    for name, sublayer in layer.named_children():
        if hasattr(sublayer, 'parameters'):
            subparams = sum(p.numel() for p in sublayer.parameters())
            print(f"      └─ {name}: {subparams:,}参数")
            
            # 进一步分析Conv2d和BatchNorm2d
            for subname, submodule in sublayer.named_children():
                if isinstance(submodule, (nn.Conv2d, nn.BatchNorm2d)):
                    submodule_params = sum(p.numel() for p in submodule.parameters())
                    print(f"         └─ {subname}: {submodule_params:,}参数")
    
    return params


def manual_calculate_parameters():
    """手动计算每个组件的理论参数量"""
    print("🔢 手动计算理论参数量:")
    print("-" * 60)
    
    # 下采样层: Conv2d(128,256,3,2,1,bias=True) + BN(256)
    downsample_conv = 128 * 256 * 3 * 3 + 256  # 294912 + 256 = 295168
    downsample_bn = 256 * 4  # weight + bias + running_mean + running_var = 1024
    downsample_total = downsample_conv + downsample_bn  # 295168 + 1024 = 296192
    print(f"   下采样层: {downsample_total:,}参数")
    print(f"      Conv2d(128,256,3,2,1,bias=True): {downsample_conv:,}")
    print(f"      BatchNorm2d(256): {downsample_bn:,}")
    
    # 残差块: Conv2d(256,256,3,1,1,bias=True) + BN(256)
    residual_conv = 256 * 256 * 3 * 3 + 256  # 589824 + 256 = 590080
    residual_bn = 256 * 4  # 1024
    residual_block = residual_conv + residual_bn  # 590080 + 1024 = 591104
    print(f"   单个残差块: {residual_block:,}参数")
    print(f"      Conv2d(256,256,3,1,1,bias=True): {residual_conv:,}")
    print(f"      BatchNorm2d(256): {residual_bn:,}")
    
    # 2个残差块
    residual_total = 2 * residual_block  # 2 * 591104 = 1182208
    print(f"   2个残差块: {residual_total:,}参数")
    
    # 1x1卷积块: Conv2d(256,256,1,1,0,bias=True) + BN(256)
    conv1x1_conv = 256 * 256 * 1 * 1 + 256  # 65536 + 256 = 65792
    conv1x1_bn = 256 * 4  # 1024
    conv1x1_block = conv1x1_conv + conv1x1_bn  # 65792 + 1024 = 66816
    print(f"   单个1x1块: {conv1x1_block:,}参数")
    print(f"      Conv2d(256,256,1,1,0,bias=True): {conv1x1_conv:,}")
    print(f"      BatchNorm2d(256): {conv1x1_bn:,}")
    
    # 5个1x1卷积块
    conv1x1_total = 5 * conv1x1_block  # 5 * 66816 = 334080
    print(f"   5个1x1块: {conv1x1_total:,}参数")
    
    # 调整层: Conv2d(256,46,1,1,0,bias=True) + BN(46) + Conv2d(46,256,1,1,0,bias=False)
    adjust_conv1 = 256 * 46 * 1 * 1 + 46  # 11776 + 46 = 11822
    adjust_bn = 46 * 4  # 184
    adjust_conv2 = 46 * 256 * 1 * 1  # 11776 (no bias)
    adjust_total = adjust_conv1 + adjust_bn + adjust_conv2  # 11822 + 184 + 11776 = 23782
    print(f"   调整层: {adjust_total:,}参数")
    print(f"      Conv2d(256,46,1,1,0,bias=True): {adjust_conv1:,}")
    print(f"      BatchNorm2d(46): {adjust_bn:,}")
    print(f"      Conv2d(46,256,1,1,0,bias=False): {adjust_conv2:,}")
    
    # SPPF层手动计算
    sppf_conv1 = 256 * 128 * 1 * 1 + 128  # 32768 + 128 = 32896
    sppf_bn1 = 128 * 4  # 512
    sppf_conv2 = 512 * 256 * 1 * 1 + 256  # 131072 + 256 = 131328
    sppf_bn2 = 256 * 4  # 1024
    sppf_total = sppf_conv1 + sppf_bn1 + sppf_conv2 + sppf_bn2  # 32896 + 512 + 131328 + 1024 = 165760
    print(f"   SPPF层: {sppf_total:,}参数")
    print(f"      cv1 Conv2d(256,128,1,bias=True): {sppf_conv1:,}")
    print(f"      cv1 BatchNorm2d(128): {sppf_bn1:,}")
    print(f"      cv2 Conv2d(512,256,1,bias=True): {sppf_conv2:,}")
    print(f"      cv2 BatchNorm2d(256): {sppf_bn2:,}")
    
    # 总计
    total_calculated = downsample_total + residual_total + conv1x1_total + adjust_total + sppf_total
    print(f"\n   理论总计: {total_calculated:,}参数")
    print(f"   目标参数: 1,990,400")
    print(f"   理论误差: {total_calculated - 1990400:,}")
    
    return {
        'downsample': downsample_total,
        'residual': residual_total,
        'conv1x1': conv1x1_total,
        'adjust': adjust_total,
        'sppf': sppf_total,
        'total': total_calculated
    }


def create_erblock5_components():
    """创建ERBlock_5的各个组件并分析参数"""
    print("\n🏗️ 创建ERBlock_5组件并分析:")
    print("-" * 60)
    
    in_ch, out_ch = 128, 256
    
    # 1. 下采样层
    downsample = nn.Sequential(
        nn.Conv2d(in_ch, out_ch, 3, 2, 1, bias=True),
        nn.BatchNorm2d(out_ch),
        nn.SiLU()
    )
    downsample_params = analyze_layer_parameters(downsample, "下采样层")
    
    # 2. 残差块
    residual_blocks = nn.Sequential()
    for i in range(2):
        residual_block = nn.Sequential(
            nn.Conv2d(out_ch, out_ch, 3, 1, 1, bias=True),
            nn.BatchNorm2d(out_ch),
            nn.SiLU()
        )
        residual_blocks.add_module(f'residual_{i}', residual_block)
    residual_params = analyze_layer_parameters(residual_blocks, "2个残差块")
    
    # 3. 1x1卷积块
    conv1x1_blocks = nn.Sequential()
    for i in range(5):
        conv1x1_block = nn.Sequential(
            nn.Conv2d(out_ch, out_ch, 1, 1, 0, bias=True),
            nn.BatchNorm2d(out_ch),
            nn.SiLU()
        )
        conv1x1_blocks.add_module(f'conv1x1_{i}', conv1x1_block)
    conv1x1_params = analyze_layer_parameters(conv1x1_blocks, "5个1x1块")
    
    # 4. 调整层
    adjust = nn.Sequential(
        nn.Conv2d(out_ch, 46, 1, 1, 0, bias=True),
        nn.BatchNorm2d(46),
        nn.Conv2d(46, out_ch, 1, 1, 0, bias=False)
    )
    adjust_params = analyze_layer_parameters(adjust, "调整层")
    
    # 5. SPPF层
    sppf = ZeroErrorSPPF(out_ch)
    sppf_params = analyze_layer_parameters(sppf, "SPPF层")
    
    # 总计
    total_actual = downsample_params + residual_params + conv1x1_params + adjust_params + sppf_params
    print(f"\n   实际总计: {total_actual:,}参数")
    print(f"   目标参数: 1,990,400")
    print(f"   实际误差: {total_actual - 1990400:,}")
    
    return {
        'downsample': downsample_params,
        'residual': residual_params,
        'conv1x1': conv1x1_params,
        'adjust': adjust_params,
        'sppf': sppf_params,
        'total': total_actual
    }


def find_error_source():
    """找出误差来源"""
    print("\n🔍 误差来源分析:")
    print("-" * 60)
    
    theoretical = manual_calculate_parameters()
    actual = create_erblock5_components()
    
    print(f"\n📊 对比分析:")
    components = ['downsample', 'residual', 'conv1x1', 'adjust', 'sppf']
    
    for comp in components:
        theory = theoretical[comp]
        actual_val = actual[comp]
        diff = actual_val - theory
        print(f"   {comp:12}: 理论{theory:8,} vs 实际{actual_val:8,} = 差{diff:6,}")
    
    total_diff = actual['total'] - theoretical['total']
    print(f"   {'总计':12}: 理论{theoretical['total']:8,} vs 实际{actual['total']:8,} = 差{total_diff:6,}")
    
    # 分析最大误差来源
    max_diff = 0
    max_comp = ""
    for comp in components:
        diff = abs(actual[comp] - theoretical[comp])
        if diff > max_diff:
            max_diff = diff
            max_comp = comp
    
    print(f"\n🎯 最大误差来源: {max_comp} ({max_diff:,}参数)")
    
    # 提供修复建议
    print(f"\n💡 修复建议:")
    if max_comp == 'adjust':
        print(f"   调整层参数过多，建议:")
        print(f"   1. 减少中间通道数 (当前46)")
        print(f"   2. 移除部分卷积层")
        print(f"   3. 使用更精确的参数计算")
    elif max_comp == 'sppf':
        print(f"   SPPF层参数不匹配，检查bias设置")
    else:
        print(f"   检查{max_comp}层的bias设置和通道数")


def main():
    """主函数"""
    print("🔍 ERBlock_5参数深度分析器")
    print("=" * 80)
    print("目标: 找出11,622参数误差的根源并提供修复方案")
    
    find_error_source()
    
    print(f"\n🎯 分析完成!")
    print("=" * 80)


if __name__ == '__main__':
    main()

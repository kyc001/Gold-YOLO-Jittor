#!/usr/bin/env python3
# -*- coding:utf-8 -*-
"""
深度分析PyTorch权重结构
基于实际权重形状重新设计Jittor架构
"""

import os
import sys
import numpy as np
from collections import defaultdict, OrderedDict
import json

def analyze_pytorch_weights():
    """深度分析PyTorch权重结构"""
    print("🔍 深度分析PyTorch权重结构")
    print("=" * 80)
    
    # 加载PyTorch权重
    weights_path = "weights/pytorch_weights.npz"
    if not os.path.exists(weights_path):
        print(f"❌ 权重文件不存在: {weights_path}")
        return None
    
    weights = np.load(weights_path)
    print(f"📊 总参数数: {len(weights)}")
    
    # 按模块分类分析
    modules = {
        'backbone': {},
        'neck': {},
        'detect': {}
    }
    
    # 分类参数
    for name, weight in weights.items():
        if 'num_batches_tracked' in name:
            continue
            
        shape = weight.shape
        size = weight.size
        
        param_info = {
            'shape': shape,
            'size': size,
            'dtype': str(weight.dtype)
        }
        
        if name.startswith('backbone.'):
            modules['backbone'][name] = param_info
        elif name.startswith('neck.'):
            modules['neck'][name] = param_info
        elif name.startswith('detect.'):
            modules['detect'][name] = param_info
    
    # 详细分析每个模块
    print(f"\n📋 模块统计:")
    for module_name, params in modules.items():
        total_params = sum(info['size'] for info in params.values())
        print(f"   {module_name}: {len(params)}个参数, {total_params:,}个元素")
    
    # 分析Neck结构 - 这是问题的关键
    print(f"\n🔍 Neck模块详细分析:")
    neck_params = modules['neck']
    
    # 按子模块分组
    neck_submodules = defaultdict(list)
    for name, info in neck_params.items():
        # 提取子模块名
        parts = name.replace('neck.', '').split('.')
        if len(parts) >= 2:
            submodule = f"{parts[0]}.{parts[1]}"
            neck_submodules[submodule].append((name, info))
        else:
            neck_submodules[parts[0]].append((name, info))
    
    for submodule, param_list in sorted(neck_submodules.items()):
        print(f"\n   📦 {submodule}:")
        for name, info in param_list:
            short_name = name.replace('neck.', '')
            print(f"      {short_name}: {info['shape']}")
    
    # 分析通道数模式
    print(f"\n🔍 通道数分析:")
    channel_patterns = analyze_channel_patterns(neck_params)
    
    for pattern, examples in channel_patterns.items():
        print(f"   {pattern}: {len(examples)}个参数")
        if len(examples) <= 3:
            for example in examples:
                print(f"      - {example}")
    
    # 生成正确的架构配置
    architecture_config = generate_architecture_config(modules)
    
    # 保存分析结果
    with open('pytorch_weights_analysis.json', 'w') as f:
        json.dump({
            'modules': {k: {name: {'shape': list(info['shape']), 'size': info['size']} 
                           for name, info in v.items()} 
                       for k, v in modules.items()},
            'architecture_config': architecture_config
        }, f, indent=2)
    
    print(f"\n💾 分析结果已保存: pytorch_weights_analysis.json")
    
    return architecture_config

def analyze_channel_patterns(neck_params):
    """分析通道数模式"""
    patterns = defaultdict(list)
    
    for name, info in neck_params.items():
        shape = info['shape']
        
        if len(shape) == 1:  # BN参数
            channels = shape[0]
            patterns[f"1D_{channels}"].append(name)
        elif len(shape) == 2:  # Linear层
            in_ch, out_ch = shape
            patterns[f"2D_{in_ch}x{out_ch}"].append(name)
        elif len(shape) == 4:  # Conv层
            out_ch, in_ch, h, w = shape
            patterns[f"4D_{in_ch}->{out_ch}_{h}x{w}"].append(name)
    
    return patterns

def generate_architecture_config(modules):
    """基于实际权重生成架构配置"""
    config = {
        'backbone': analyze_backbone_config(modules['backbone']),
        'neck': analyze_neck_config(modules['neck']),
        'detect': analyze_detect_config(modules['detect'])
    }
    
    return config

def analyze_backbone_config(backbone_params):
    """分析Backbone配置"""
    config = {
        'stem': {},
        'ERBlocks': {}
    }
    
    # 分析stem
    for name, info in backbone_params.items():
        if 'stem' in name:
            if 'conv.weight' in name:
                config['stem']['conv_channels'] = info['shape']
            elif 'bn.weight' in name:
                config['stem']['bn_channels'] = info['shape'][0]
    
    # 分析ERBlocks
    erblock_pattern = {}
    for name, info in backbone_params.items():
        if 'ERBlock' in name:
            # 提取ERBlock信息
            import re
            match = re.search(r'ERBlock_(\d+)\.(\d+)', name)
            if match:
                stage = int(match.group(1))
                block = int(match.group(2))
                
                if stage not in erblock_pattern:
                    erblock_pattern[stage] = {}
                if block not in erblock_pattern[stage]:
                    erblock_pattern[stage][block] = {}
                
                if 'conv.weight' in name:
                    erblock_pattern[stage][block]['conv_shape'] = info['shape']
                elif 'bn.weight' in name:
                    erblock_pattern[stage][block]['bn_channels'] = info['shape'][0]
    
    config['ERBlocks'] = erblock_pattern
    return config

def analyze_neck_config(neck_params):
    """分析Neck配置 - 关键部分"""
    config = {
        'low_IFM': {},
        'high_IFM': {},
        'LAF': {},
        'Inject': {},
        'Rep': {}
    }
    
    # 分析low_IFM
    low_ifm_info = {}
    for name, info in neck_params.items():
        if 'low_IFM' in name:
            if 'conv.weight' in name and 'low_IFM.0.conv.weight' in name:
                # 这是关键的输入通道信息
                out_ch, in_ch, h, w = info['shape']
                low_ifm_info['input_channels'] = in_ch  # 应该是backbone输出通道
                low_ifm_info['output_channels'] = out_ch
            elif 'bn.weight' in name and 'low_IFM.0.bn.weight' in name:
                low_ifm_info['bn_channels'] = info['shape'][0]
    
    config['low_IFM'] = low_ifm_info
    
    # 分析high_IFM transformer结构
    transformer_info = {}
    for name, info in neck_params.items():
        if 'high_IFM.transformer_blocks' in name:
            if 'to_q.c.weight' in name:
                # 提取transformer的实际通道数
                if len(info['shape']) == 4:
                    out_ch, in_ch, h, w = info['shape']
                    transformer_info['q_channels'] = out_ch
                    transformer_info['input_channels'] = in_ch
                elif len(info['shape']) == 2:
                    out_ch, in_ch = info['shape']
                    transformer_info['q_channels'] = out_ch
                    transformer_info['input_channels'] = in_ch
                break
    
    config['high_IFM'] = transformer_info
    
    return config

def analyze_detect_config(detect_params):
    """分析检测头配置"""
    config = {
        'stems': {},
        'cls_convs': {},
        'reg_convs': {},
        'cls_preds': {},
        'reg_preds': {}
    }
    
    # 分析各个组件的通道数
    for name, info in detect_params.items():
        if 'stems' in name and 'conv.weight' in name:
            # 提取stems的通道信息
            level = name.split('.')[1]  # stems.0, stems.1, stems.2
            config['stems'][level] = info['shape']
        elif 'cls_preds' in name and 'weight' in name:
            level = name.split('.')[1]
            config['cls_preds'][level] = info['shape']
        elif 'reg_preds' in name and 'weight' in name:
            level = name.split('.')[1]
            config['reg_preds'][level] = info['shape']
    
    return config

def main():
    """主函数"""
    config = analyze_pytorch_weights()
    
    if config:
        print(f"\n🎯 关键发现:")
        print(f"   • Backbone输出通道: 分析完成")
        print(f"   • Neck输入通道: {config['neck']['low_IFM'].get('input_channels', 'Unknown')}")
        print(f"   • Neck输出通道: {config['neck']['low_IFM'].get('output_channels', 'Unknown')}")
        print(f"   • Transformer通道: {config['neck']['high_IFM'].get('input_channels', 'Unknown')}")
        
        print(f"\n🚀 下一步:")
        print(f"   1. 基于实际权重形状重新设计Jittor架构")
        print(f"   2. 确保每个参数的形状完全匹配")
        print(f"   3. 重新进行权重转换和测试")

if __name__ == '__main__':
    main()

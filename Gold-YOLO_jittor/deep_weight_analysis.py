#!/usr/bin/env python3
# -*- coding:utf-8 -*-
"""
深度权重分析器
严格分析PyTorch权重结构，为Jittor架构对齐提供精确指导
"""

import os
import sys
import numpy as np
from collections import OrderedDict, defaultdict
import json
import re

def deep_analyze_pytorch_weights():
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
    
    # 详细分析每个模块
    analysis = {
        'backbone': analyze_backbone_weights(weights),
        'neck': analyze_neck_weights(weights),
        'detect': analyze_detect_weights(weights),
        'architecture_map': create_architecture_map(weights)
    }
    
    # 保存详细分析结果
    with open('deep_weight_analysis.json', 'w') as f:
        json.dump(analysis, f, indent=2, default=convert_numpy)
    
    print(f"\n💾 深度分析结果已保存: deep_weight_analysis.json")
    
    return analysis

def convert_numpy(obj):
    """转换numpy对象为JSON可序列化格式"""
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, tuple):
        return list(obj)
    return obj

def analyze_backbone_weights(weights):
    """深度分析Backbone权重"""
    print(f"\n🔍 深度分析Backbone权重")
    print("-" * 60)
    
    backbone_weights = {k: v for k, v in weights.items() if k.startswith('backbone.')}
    
    analysis = {
        'total_params': len(backbone_weights),
        'total_elements': sum(w.size for w in backbone_weights.values()),
        'structure': {},
        'channel_flow': {},
        'layer_details': {}
    }
    
    # 分析stem结构
    stem_analysis = analyze_stem_structure(backbone_weights)
    analysis['structure']['stem'] = stem_analysis
    
    # 分析ERBlock结构
    erblock_analysis = analyze_erblock_structure(backbone_weights)
    analysis['structure']['erblocks'] = erblock_analysis
    
    # 分析通道流
    channel_flow = trace_backbone_channel_flow(backbone_weights)
    analysis['channel_flow'] = channel_flow
    
    print(f"   Backbone参数: {analysis['total_params']}")
    print(f"   Backbone元素: {analysis['total_elements']:,}")
    
    return analysis

def analyze_stem_structure(backbone_weights):
    """分析stem结构"""
    stem_weights = {k: v for k, v in backbone_weights.items() if 'stem' in k}
    
    stem_analysis = {
        'conv_weight': None,
        'bn_params': {},
        'input_channels': None,
        'output_channels': None
    }
    
    for name, weight in stem_weights.items():
        if 'conv.weight' in name:
            stem_analysis['conv_weight'] = weight.shape
            stem_analysis['output_channels'] = weight.shape[0]
            stem_analysis['input_channels'] = weight.shape[1]
        elif 'bn.weight' in name:
            stem_analysis['bn_params']['weight'] = weight.shape
        elif 'bn.bias' in name:
            stem_analysis['bn_params']['bias'] = weight.shape
        elif 'bn.running_mean' in name:
            stem_analysis['bn_params']['running_mean'] = weight.shape
        elif 'bn.running_var' in name:
            stem_analysis['bn_params']['running_var'] = weight.shape
    
    print(f"   Stem: {stem_analysis['input_channels']} -> {stem_analysis['output_channels']}")
    
    return stem_analysis

def analyze_erblock_structure(backbone_weights):
    """分析ERBlock结构"""
    erblock_weights = {k: v for k, v in backbone_weights.items() if 'ERBlock' in k}
    
    # 按ERBlock分组
    erblock_groups = defaultdict(lambda: defaultdict(list))
    
    for name, weight in erblock_weights.items():
        # 提取ERBlock信息: ERBlock_X.Y
        match = re.search(r'ERBlock_(\d+)\.(\d+)', name)
        if match:
            stage = int(match.group(1))
            block = int(match.group(2))
            erblock_groups[stage][block].append((name, weight))
    
    erblock_analysis = {}
    
    for stage in sorted(erblock_groups.keys()):
        stage_analysis = {}
        
        for block in sorted(erblock_groups[stage].keys()):
            block_params = erblock_groups[stage][block]
            block_analysis = analyze_single_erblock(block_params)
            stage_analysis[f'block_{block}'] = block_analysis
        
        erblock_analysis[f'stage_{stage}'] = stage_analysis
        
        # 打印stage信息
        total_blocks = len(stage_analysis)
        print(f"   ERBlock_Stage_{stage}: {total_blocks}个子块")
    
    return erblock_analysis

def analyze_single_erblock(block_params):
    """分析单个ERBlock"""
    block_analysis = {
        'conv_layers': [],
        'bn_layers': [],
        'special_layers': [],
        'input_channels': None,
        'output_channels': None
    }
    
    for name, weight in block_params:
        if 'conv.weight' in name:
            conv_info = {
                'name': name,
                'shape': weight.shape,
                'out_ch': weight.shape[0],
                'in_ch': weight.shape[1],
                'kernel': (weight.shape[2], weight.shape[3]) if len(weight.shape) == 4 else None
            }
            block_analysis['conv_layers'].append(conv_info)
            
            # 记录输入输出通道
            if block_analysis['input_channels'] is None:
                block_analysis['input_channels'] = weight.shape[1]
            block_analysis['output_channels'] = weight.shape[0]
            
        elif 'bn.' in name:
            bn_info = {
                'name': name,
                'shape': weight.shape,
                'channels': weight.shape[0] if len(weight.shape) == 1 else None
            }
            block_analysis['bn_layers'].append(bn_info)
        else:
            block_analysis['special_layers'].append({
                'name': name,
                'shape': weight.shape
            })
    
    return block_analysis

def trace_backbone_channel_flow(backbone_weights):
    """追踪backbone的通道流"""
    channel_flow = {
        'stem': None,
        'erblock_stages': {}
    }
    
    # 分析stem输出
    for name, weight in backbone_weights.items():
        if 'stem.block.conv.weight' in name:
            channel_flow['stem'] = {
                'input': weight.shape[1],
                'output': weight.shape[0]
            }
            break
    
    # 分析每个ERBlock stage的输入输出
    erblock_weights = {k: v for k, v in backbone_weights.items() if 'ERBlock' in k and 'conv.weight' in k}
    
    stage_channels = {}
    for name, weight in erblock_weights.items():
        match = re.search(r'ERBlock_(\d+)', name)
        if match:
            stage = int(match.group(1))
            if stage not in stage_channels:
                stage_channels[stage] = {
                    'input': weight.shape[1],
                    'output': weight.shape[0]
                }
            else:
                # 更新输出通道（最后一层的输出）
                stage_channels[stage]['output'] = weight.shape[0]
    
    channel_flow['erblock_stages'] = stage_channels
    
    return channel_flow

def analyze_neck_weights(weights):
    """深度分析Neck权重"""
    print(f"\n🔍 深度分析Neck权重")
    print("-" * 60)
    
    neck_weights = {k: v for k, v in weights.items() if k.startswith('neck.')}
    
    analysis = {
        'total_params': len(neck_weights),
        'total_elements': sum(w.size for w in neck_weights.values()),
        'modules': {},
        'channel_flow': {},
        'transformer_details': {}
    }
    
    # 分析各个子模块
    modules = ['low_IFM', 'high_IFM', 'LAF_p3', 'LAF_p4', 'Inject_p3', 'Inject_p4', 
               'Inject_n4', 'Inject_n5', 'Rep_p3', 'Rep_p4', 'Rep_n4', 'Rep_n5',
               'reduce_layer_p4', 'reduce_layer_c5', 'conv_1x1_n']
    
    for module in modules:
        module_weights = {k: v for k, v in neck_weights.items() if module in k}
        if module_weights:
            module_analysis = analyze_neck_module(module, module_weights)
            analysis['modules'][module] = module_analysis
    
    # 特别分析transformer结构
    transformer_analysis = analyze_transformer_structure(neck_weights)
    analysis['transformer_details'] = transformer_analysis
    
    # 分析neck的通道流
    neck_channel_flow = trace_neck_channel_flow(neck_weights)
    analysis['channel_flow'] = neck_channel_flow
    
    print(f"   Neck参数: {analysis['total_params']}")
    print(f"   Neck元素: {analysis['total_elements']:,}")
    print(f"   Neck模块数: {len(analysis['modules'])}")
    
    return analysis

def analyze_neck_module(module_name, module_weights):
    """分析neck模块"""
    module_analysis = {
        'param_count': len(module_weights),
        'layers': [],
        'input_channels': None,
        'output_channels': None,
        'special_structure': {}
    }
    
    for name, weight in module_weights.items():
        layer_info = {
            'name': name.replace(f'neck.{module_name}.', ''),
            'full_name': name,
            'shape': weight.shape,
            'type': classify_layer_type(name, weight)
        }
        
        # 分析通道信息
        if 'conv.weight' in name and len(weight.shape) == 4:
            layer_info['out_channels'] = weight.shape[0]
            layer_info['in_channels'] = weight.shape[1]
            layer_info['kernel_size'] = (weight.shape[2], weight.shape[3])
            
            if module_analysis['input_channels'] is None:
                module_analysis['input_channels'] = weight.shape[1]
            module_analysis['output_channels'] = weight.shape[0]
        
        module_analysis['layers'].append(layer_info)
    
    return module_analysis

def classify_layer_type(name, weight):
    """分类层类型"""
    if 'conv.weight' in name:
        if len(weight.shape) == 4:
            return f"Conv2d_{weight.shape[2]}x{weight.shape[3]}"
        else:
            return "Conv_other"
    elif 'bn.weight' in name:
        return "BatchNorm_weight"
    elif 'bn.bias' in name:
        return "BatchNorm_bias"
    elif 'bn.running_mean' in name:
        return "BatchNorm_running_mean"
    elif 'bn.running_var' in name:
        return "BatchNorm_running_var"
    elif 'bias' in name:
        return "Bias"
    else:
        return "Other"

def analyze_transformer_structure(neck_weights):
    """分析transformer结构"""
    transformer_weights = {k: v for k, v in neck_weights.items() if 'transformer_blocks' in k}
    
    transformer_analysis = {
        'num_blocks': 0,
        'attention_structure': {},
        'mlp_structure': {},
        'channel_details': {}
    }
    
    # 统计transformer blocks数量
    block_indices = set()
    for name in transformer_weights.keys():
        match = re.search(r'transformer_blocks\.(\d+)', name)
        if match:
            block_indices.add(int(match.group(1)))
    
    transformer_analysis['num_blocks'] = len(block_indices)
    
    # 分析attention结构
    attention_weights = {k: v for k, v in transformer_weights.items() if 'attn.' in k}
    transformer_analysis['attention_structure'] = analyze_attention_structure(attention_weights)
    
    # 分析MLP结构
    mlp_weights = {k: v for k, v in transformer_weights.items() if 'mlp.' in k}
    transformer_analysis['mlp_structure'] = analyze_mlp_structure(mlp_weights)
    
    print(f"   Transformer blocks: {transformer_analysis['num_blocks']}")
    
    return transformer_analysis

def analyze_attention_structure(attention_weights):
    """分析attention结构"""
    attention_analysis = {
        'to_q': None,
        'to_k': None,
        'to_v': None,
        'proj': None
    }
    
    for name, weight in attention_weights.items():
        if 'to_q.c.weight' in name:
            attention_analysis['to_q'] = {
                'shape': weight.shape,
                'out_channels': weight.shape[0],
                'in_channels': weight.shape[1]
            }
        elif 'to_k.c.weight' in name:
            attention_analysis['to_k'] = {
                'shape': weight.shape,
                'out_channels': weight.shape[0],
                'in_channels': weight.shape[1]
            }
        elif 'to_v.c.weight' in name:
            attention_analysis['to_v'] = {
                'shape': weight.shape,
                'out_channels': weight.shape[0],
                'in_channels': weight.shape[1]
            }
        elif 'proj.1.c.weight' in name:
            attention_analysis['proj'] = {
                'shape': weight.shape,
                'out_channels': weight.shape[0],
                'in_channels': weight.shape[1]
            }
    
    return attention_analysis

def analyze_mlp_structure(mlp_weights):
    """分析MLP结构"""
    mlp_analysis = {
        'fc1': None,
        'dwconv': None,
        'fc2': None
    }
    
    for name, weight in mlp_weights.items():
        if 'fc1.c.weight' in name:
            mlp_analysis['fc1'] = {
                'shape': weight.shape,
                'out_channels': weight.shape[0],
                'in_channels': weight.shape[1]
            }
        elif 'dwconv.weight' in name:
            mlp_analysis['dwconv'] = {
                'shape': weight.shape,
                'groups': weight.shape[0],  # depthwise conv
                'kernel_size': (weight.shape[2], weight.shape[3])
            }
        elif 'fc2.c.weight' in name:
            mlp_analysis['fc2'] = {
                'shape': weight.shape,
                'out_channels': weight.shape[0],
                'in_channels': weight.shape[1]
            }
    
    return mlp_analysis

def trace_neck_channel_flow(neck_weights):
    """追踪neck的通道流"""
    channel_flow = {
        'input_to_low_IFM': None,
        'low_IFM_output': None,
        'transformer_channels': None,
        'final_output': None
    }
    
    # 分析low_IFM输入
    for name, weight in neck_weights.items():
        if 'low_IFM.0.conv.weight' in name:
            channel_flow['input_to_low_IFM'] = {
                'input_channels': weight.shape[1],
                'output_channels': weight.shape[0]
            }
            break
    
    # 分析transformer通道
    for name, weight in neck_weights.items():
        if 'transformer_blocks.0.attn.to_q.c.weight' in name:
            channel_flow['transformer_channels'] = {
                'input_channels': weight.shape[1],
                'q_channels': weight.shape[0]
            }
            break
    
    # 分析最终输出
    for name, weight in neck_weights.items():
        if 'conv_1x1_n.weight' in name:
            channel_flow['final_output'] = {
                'input_channels': weight.shape[1],
                'output_channels': weight.shape[0]
            }
            break
    
    return channel_flow

def analyze_detect_weights(weights):
    """深度分析检测头权重"""
    print(f"\n🔍 深度分析检测头权重")
    print("-" * 60)
    
    detect_weights = {k: v for k, v in weights.items() if k.startswith('detect.')}
    
    analysis = {
        'total_params': len(detect_weights),
        'total_elements': sum(w.size for w in detect_weights.values()),
        'stems': {},
        'cls_convs': {},
        'reg_convs': {},
        'cls_preds': {},
        'reg_preds': {},
        'scales': []
    }
    
    # 分析各个组件
    components = ['stems', 'cls_convs', 'reg_convs', 'cls_preds', 'reg_preds']
    
    for component in components:
        component_weights = {k: v for k, v in detect_weights.items() if component in k}
        component_analysis = analyze_detect_component(component, component_weights)
        analysis[component] = component_analysis
    
    # 分析尺度信息
    scales = set()
    for name in detect_weights.keys():
        match = re.search(r'\.(0|1|2)\.', name)
        if match:
            scales.add(int(match.group(1)))
    
    analysis['scales'] = sorted(list(scales))
    
    print(f"   检测头参数: {analysis['total_params']}")
    print(f"   检测头元素: {analysis['total_elements']:,}")
    print(f"   检测尺度: {analysis['scales']}")
    
    return analysis

def analyze_detect_component(component_name, component_weights):
    """分析检测头组件"""
    component_analysis = {
        'scales': {},
        'total_params': len(component_weights)
    }
    
    # 按尺度分组
    scale_groups = defaultdict(list)
    
    for name, weight in component_weights.items():
        match = re.search(r'\.(0|1|2)\.', name)
        if match:
            scale = int(match.group(1))
            scale_groups[scale].append((name, weight))
    
    for scale in sorted(scale_groups.keys()):
        scale_params = scale_groups[scale]
        scale_analysis = {
            'param_count': len(scale_params),
            'layers': []
        }
        
        for name, weight in scale_params:
            layer_info = {
                'name': name,
                'shape': weight.shape,
                'type': classify_layer_type(name, weight)
            }
            
            if 'weight' in name and len(weight.shape) >= 2:
                if len(weight.shape) == 4:  # Conv2d
                    layer_info['out_channels'] = weight.shape[0]
                    layer_info['in_channels'] = weight.shape[1]
                elif len(weight.shape) == 2:  # Linear
                    layer_info['out_features'] = weight.shape[0]
                    layer_info['in_features'] = weight.shape[1]
            
            scale_analysis['layers'].append(layer_info)
        
        component_analysis['scales'][scale] = scale_analysis
    
    return component_analysis

def create_architecture_map(weights):
    """创建架构映射"""
    architecture_map = {
        'parameter_count_by_module': {},
        'channel_progression': {},
        'critical_connections': {}
    }
    
    # 统计各模块参数数量
    modules = ['backbone', 'neck', 'detect']
    for module in modules:
        module_weights = {k: v for k, v in weights.items() if k.startswith(f'{module}.')}
        architecture_map['parameter_count_by_module'][module] = {
            'param_count': len(module_weights),
            'element_count': sum(w.size for w in module_weights.values())
        }
    
    # 分析通道进展
    channel_progression = trace_overall_channel_progression(weights)
    architecture_map['channel_progression'] = channel_progression
    
    return architecture_map

def trace_overall_channel_progression(weights):
    """追踪整体通道进展"""
    progression = {
        'input': 3,  # RGB输入
        'backbone_stages': [],
        'neck_input': None,
        'neck_output': None,
        'detect_inputs': []
    }
    
    # 追踪backbone stages
    backbone_weights = {k: v for k, v in weights.items() if k.startswith('backbone.') and 'conv.weight' in k}
    
    # 按ERBlock stage排序
    stage_outputs = {}
    for name, weight in backbone_weights.items():
        if 'ERBlock' in name:
            match = re.search(r'ERBlock_(\d+)', name)
            if match:
                stage = int(match.group(1))
                stage_outputs[stage] = weight.shape[0]  # 输出通道
    
    for stage in sorted(stage_outputs.keys()):
        progression['backbone_stages'].append(stage_outputs[stage])
    
    # 追踪neck输入输出
    for name, weight in weights.items():
        if 'neck.low_IFM.0.conv.weight' in name:
            progression['neck_input'] = weight.shape[1]
        elif 'neck.conv_1x1_n.weight' in name:
            progression['neck_output'] = weight.shape[0]
    
    # 追踪detect输入
    detect_weights = {k: v for k, v in weights.items() if k.startswith('detect.stems') and 'conv.weight' in k}
    
    scale_inputs = {}
    for name, weight in detect_weights.items():
        match = re.search(r'stems\.(\d+)\.conv\.weight', name)
        if match:
            scale = int(match.group(1))
            scale_inputs[scale] = weight.shape[1]  # 输入通道
    
    for scale in sorted(scale_inputs.keys()):
        progression['detect_inputs'].append(scale_inputs[scale])
    
    return progression

def main():
    """主函数"""
    print("🔍 Gold-YOLO深度权重分析系统")
    print("=" * 80)
    
    analysis = deep_analyze_pytorch_weights()
    
    if analysis:
        print(f"\n🎯 关键发现:")
        
        # Backbone分析
        backbone = analysis['backbone']
        print(f"   📦 Backbone:")
        print(f"      • 参数数: {backbone['total_params']}")
        print(f"      • 通道流: {backbone['channel_flow']}")
        
        # Neck分析
        neck = analysis['neck']
        print(f"   📦 Neck:")
        print(f"      • 参数数: {neck['total_params']}")
        print(f"      • 模块数: {len(neck['modules'])}")
        print(f"      • Transformer blocks: {neck['transformer_details']['num_blocks']}")
        
        # Detect分析
        detect = analysis['detect']
        print(f"   📦 Detect:")
        print(f"      • 参数数: {detect['total_params']}")
        print(f"      • 检测尺度: {detect['scales']}")
        
        # 整体架构
        arch_map = analysis['architecture_map']
        print(f"   📊 整体通道进展:")
        print(f"      • Backbone stages: {arch_map['channel_progression']['backbone_stages']}")
        print(f"      • Neck输入: {arch_map['channel_progression']['neck_input']}")
        print(f"      • Detect输入: {arch_map['channel_progression']['detect_inputs']}")
        
        print(f"\n🚀 下一步:")
        print(f"   1. 基于此分析创建严格对齐的Jittor架构")
        print(f"   2. 确保每个参数形状完全匹配")
        print(f"   3. 实现100%权重转换成功率")

if __name__ == '__main__':
    main()

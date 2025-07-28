#!/usr/bin/env python3
"""
对比PyTorch版本的模型初始化权重
"""

import torch
import numpy as np
import sys
import os

# 添加路径
sys.path.append('.')
sys.path.append('..')

def test_pytorch_model_initialization():
    """对比PyTorch模型初始化权重"""
    print("🔍 开始检查PyTorch模型初始化权重...")
    
    try:
        # 导入PyTorch版本的模型
        from yolov6.models.yolo import Model as PyTorchModel
        from yolov6.utils.config import Config
        
        # 加载配置
        config = Config.fromfile('configs/gold_yolo-n.py')
        
        # 创建PyTorch模型
        pytorch_model = PyTorchModel(config, channels=3, num_classes=20)
        
        print(f"✅ PyTorch模型创建成功")

        # 检查关键参数
        print(f"\n📊 模型关键参数:")
        if hasattr(pytorch_model, 'detect'):
            print(f"   use_dfl: {pytorch_model.detect.use_dfl}")
            print(f"   reg_max: {pytorch_model.detect.reg_max}")
            if hasattr(pytorch_model.detect, 'proj_conv'):
                print(f"   proj_conv形状: {pytorch_model.detect.proj_conv.weight.shape}")
                print(f"   proj_conv权重均值: {float(pytorch_model.detect.proj_conv.weight.mean().data):.6f}")
                print(f"   proj_conv权重标准差: {float(pytorch_model.detect.proj_conv.weight.std().data):.6f}")
        
        # 检查关键层的初始化
        print(f"\n📊 关键层权重统计:")
        
        # 检查backbone的第一层
        first_conv = None
        for name, module in pytorch_model.named_modules():
            if 'conv' in name.lower() and hasattr(module, 'weight'):
                first_conv = module
                print(f"   第一个卷积层: {name}")
                print(f"     权重形状: {module.weight.shape}")
                print(f"     权重均值: {float(module.weight.mean().data):.6f}")
                print(f"     权重标准差: {float(module.weight.std().data):.6f}")
                print(f"     权重范围: [{float(module.weight.min().data):.6f}, {float(module.weight.max().data):.6f}]")
                break
        
        # 检查Head层的初始化 - 更详细的搜索
        head_layers = []
        detect_layers = []
        cls_layers = []
        reg_layers = []
        
        for name, module in pytorch_model.named_modules():
            if hasattr(module, 'weight'):
                if 'head' in name.lower():
                    head_layers.append((name, module))
                elif 'detect' in name.lower():
                    detect_layers.append((name, module))
                elif 'cls' in name.lower():
                    cls_layers.append((name, module))
                elif 'reg' in name.lower():
                    reg_layers.append((name, module))
        
        print(f"\n📊 Head相关层权重统计:")
        print(f"   Head层: {len(head_layers)}层")
        print(f"   Detect层: {len(detect_layers)}层")
        print(f"   Cls层: {len(cls_layers)}层")
        print(f"   Reg层: {len(reg_layers)}层")
        
        # 显示所有相关层
        all_head_layers = head_layers + detect_layers + cls_layers + reg_layers
        for i, (name, module) in enumerate(all_head_layers[:5]):  # 显示前5层
            print(f"   层{i+1}: {name}")
            print(f"     权重形状: {module.weight.shape}")
            print(f"     权重均值: {float(module.weight.mean().data):.6f}")
            print(f"     权重标准差: {float(module.weight.std().data):.6f}")
            if hasattr(module, 'bias') and module.bias is not None:
                print(f"     偏置均值: {float(module.bias.mean().data):.6f}")
                print(f"     偏置范围: [{float(module.bias.min().data):.6f}, {float(module.bias.max().data):.6f}]")
            else:
                print(f"     无偏置")
        
        # 检查BatchNorm层
        bn_layers = []
        for name, module in pytorch_model.named_modules():
            if 'bn' in name.lower() or 'norm' in name.lower():
                if hasattr(module, 'weight'):
                    bn_layers.append((name, module))
        
        print(f"\n📊 BatchNorm层统计 (共{len(bn_layers)}层):")
        for i, (name, module) in enumerate(bn_layers[:3]):  # 只显示前3层
            print(f"   BN层{i+1}: {name}")
            if hasattr(module, 'weight') and module.weight is not None:
                print(f"     权重均值: {float(module.weight.mean().data):.6f}")
                print(f"     权重标准差: {float(module.weight.std().data):.6f}")
            if hasattr(module, 'bias') and module.bias is not None:
                print(f"     偏置均值: {float(module.bias.mean().data):.6f}")
        
        # 测试前向传播
        print(f"\n🧪 测试前向传播:")
        test_input = torch.randn((1, 3, 640, 640))
        print(f"   输入形状: {test_input.shape}")
        print(f"   输入数值范围: [{float(test_input.min().data):.6f}, {float(test_input.max().data):.6f}]")
        
        with torch.no_grad():
            outputs = pytorch_model(test_input)
            
        if isinstance(outputs, (list, tuple)):
            print(f"   输出类型: {type(outputs)}, 长度: {len(outputs)}")
            for i, output in enumerate(outputs):
                if hasattr(output, 'shape'):
                    print(f"   输出{i}形状: {output.shape}")
                    print(f"   输出{i}数值范围: [{float(output.min().data):.6f}, {float(output.max().data):.6f}]")
                elif isinstance(output, (list, tuple)):
                    print(f"   输出{i}是列表，长度: {len(output)}")
                    for j, sub_output in enumerate(output):
                        if hasattr(sub_output, 'shape'):
                            print(f"     子输出{j}形状: {sub_output.shape}")
                            print(f"     子输出{j}数值范围: [{float(sub_output.min().data):.6f}, {float(sub_output.max().data):.6f}]")
        else:
            print(f"   输出形状: {outputs.shape}")
            print(f"   输出数值范围: [{float(outputs.min().data):.6f}, {float(outputs.max().data):.6f}]")
        
        return True
        
    except Exception as e:
        print(f"❌ PyTorch模型初始化测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    test_pytorch_model_initialization()

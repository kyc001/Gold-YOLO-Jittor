#!/usr/bin/env python3
"""
深入分析PyTorch版本的详细信息
细致到每一个通道和接口
"""

import os
import sys
import torch
import torch.nn as nn
from pathlib import Path

# 添加PyTorch版本路径
pytorch_path = "/home/kyc/project/GOLD-YOLO/Gold-YOLO_pytorch"
sys.path.insert(0, pytorch_path)

def analyze_pytorch_model():
    """深入分析PyTorch版本模型"""
    print("🔍 深入分析PyTorch版本GOLD-YOLO")
    print("=" * 80)
    
    try:
        # 导入PyTorch版本
        from yolov6.models.yolo import build_model
        from yolov6.utils.config import Config
        
        # 加载配置
        config_path = os.path.join(pytorch_path, "configs/gold_yolo-n.py")
        cfg = Config.fromfile(config_path)
        
        print(f"📁 配置文件: {config_path}")
        print(f"🎯 模型配置: {cfg.model}")
        
        # 创建模型
        model = build_model(cfg, num_classes=20, device='cpu')
        model.eval()
        
        print(f"\n📊 PyTorch模型分析:")
        print(f"   模型类型: {type(model)}")
        
        # 分析模型结构
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        print(f"   总参数: {total_params:,} ({total_params/1e6:.2f}M)")
        print(f"   可训练参数: {trainable_params:,} ({trainable_params/1e6:.2f}M)")
        
        # 分析各部分参数
        backbone_params = sum(p.numel() for p in model.backbone.parameters())
        neck_params = sum(p.numel() for p in model.neck.parameters()) if hasattr(model, 'neck') else 0
        head_params = sum(p.numel() for p in model.head.parameters()) if hasattr(model, 'head') else 0
        
        print(f"   Backbone参数: {backbone_params:,} ({backbone_params/total_params*100:.1f}%)")
        print(f"   Neck参数: {neck_params:,} ({neck_params/total_params*100:.1f}%)")
        print(f"   Head参数: {head_params:,} ({head_params/total_params*100:.1f}%)")
        
        # 测试前向传播
        print(f"\n🔄 测试前向传播:")
        test_input = torch.randn(1, 3, 500, 500)
        print(f"   输入形状: {test_input.shape}")
        
        with torch.no_grad():
            outputs = model(test_input)
        
        print(f"   输出类型: {type(outputs)}")
        if isinstance(outputs, (list, tuple)):
            for i, output in enumerate(outputs):
                if hasattr(output, 'shape'):
                    print(f"     输出{i}: {output.shape}")
                else:
                    print(f"     输出{i}: {type(output)}")
        else:
            print(f"   输出形状: {outputs.shape}")
        
        # 分析backbone详细结构
        print(f"\n🏗️ Backbone详细分析:")
        print(f"   Backbone类型: {type(model.backbone)}")
        
        # 逐层分析backbone
        for name, module in model.backbone.named_children():
            if hasattr(module, 'weight'):
                print(f"     {name}: {module.weight.shape}")
            elif hasattr(module, '__len__'):
                print(f"     {name}: {len(module)}层")
                for i, sub_module in enumerate(module):
                    if hasattr(sub_module, 'weight'):
                        print(f"       {name}[{i}]: {sub_module.weight.shape}")
            else:
                print(f"     {name}: {type(module)}")
        
        # 分析neck详细结构
        if hasattr(model, 'neck'):
            print(f"\n🔗 Neck详细分析:")
            print(f"   Neck类型: {type(model.neck)}")
            
            for name, module in model.neck.named_children():
                print(f"     {name}: {type(module)}")
        
        # 分析head详细结构
        if hasattr(model, 'head'):
            print(f"\n🎯 Head详细分析:")
            print(f"   Head类型: {type(model.head)}")
            
            # 检查head的关键属性
            if hasattr(model.head, 'use_dfl'):
                print(f"     use_dfl: {model.head.use_dfl}")
            if hasattr(model.head, 'reg_max'):
                print(f"     reg_max: {model.head.reg_max}")
            if hasattr(model.head, 'nc'):
                print(f"     num_classes: {model.head.nc}")
            if hasattr(model.head, 'nl'):
                print(f"     num_layers: {model.head.nl}")
            
            # 分析head的各个组件
            for name, module in model.head.named_children():
                if hasattr(module, '__len__'):
                    print(f"     {name}: {len(module)}个组件")
                    for i, sub_module in enumerate(module):
                        if hasattr(sub_module, 'weight'):
                            print(f"       {name}[{i}]: {sub_module.weight.shape}")
                        else:
                            print(f"       {name}[{i}]: {type(sub_module)}")
                else:
                    print(f"     {name}: {type(module)}")
        
        # 分析特征图尺寸
        print(f"\n📐 特征图尺寸分析:")
        
        # Hook函数来捕获中间特征
        feature_shapes = {}
        
        def hook_fn(name):
            def hook(module, input, output):
                if isinstance(output, torch.Tensor):
                    feature_shapes[name] = output.shape
                elif isinstance(output, (list, tuple)):
                    for i, out in enumerate(output):
                        if isinstance(out, torch.Tensor):
                            feature_shapes[f"{name}_out{i}"] = out.shape
            return hook
        
        # 注册hook
        hooks = []
        if hasattr(model, 'backbone'):
            hook = model.backbone.register_forward_hook(hook_fn('backbone'))
            hooks.append(hook)
        if hasattr(model, 'neck'):
            hook = model.neck.register_forward_hook(hook_fn('neck'))
            hooks.append(hook)
        if hasattr(model, 'head'):
            hook = model.head.register_forward_hook(hook_fn('head'))
            hooks.append(hook)
        
        # 重新前向传播
        with torch.no_grad():
            _ = model(test_input)
        
        # 移除hook
        for hook in hooks:
            hook.remove()
        
        # 显示特征图尺寸
        for name, shape in feature_shapes.items():
            print(f"     {name}: {shape}")
        
        # 计算anchor数量
        print(f"\n⚓ Anchor数量分析:")
        if hasattr(model, 'head') and hasattr(model.head, 'stride'):
            strides = model.head.stride
            print(f"   Strides: {strides}")
            
            input_size = 500
            total_anchors = 0
            for stride in strides:
                feature_size = input_size // stride
                anchors = feature_size * feature_size
                total_anchors += anchors
                print(f"     Stride {stride}: {feature_size}x{feature_size} = {anchors} anchors")
            
            print(f"   总Anchor数: {total_anchors}")
        
        return {
            'model': model,
            'total_params': total_params,
            'trainable_params': trainable_params,
            'backbone_params': backbone_params,
            'neck_params': neck_params,
            'head_params': head_params,
            'feature_shapes': feature_shapes,
            'outputs': outputs
        }
        
    except Exception as e:
        print(f"❌ 分析失败: {e}")
        import traceback
        traceback.print_exc()
        return None

def compare_with_jittor():
    """对比Jittor版本"""
    print(f"\n🔄 对比Jittor版本:")
    print("=" * 80)
    
    # 这里可以加载Jittor版本进行对比
    # 暂时先输出Jittor版本的已知信息
    print(f"Jittor版本信息:")
    print(f"   总参数: 5,711,613 (5.71M)")
    print(f"   输出形状: feats(list), [1,5249,20], [1,5249,68]")
    print(f"   Anchor数: 5249")
    print(f"   梯度问题: 476/480参数梯度为零")

def main():
    print("🔍 PyTorch版本深入分析")
    print("=" * 80)
    
    # 分析PyTorch版本
    pytorch_info = analyze_pytorch_model()
    
    if pytorch_info:
        print(f"\n✅ PyTorch版本分析完成!")
        
        # 对比Jittor版本
        compare_with_jittor()
        
        # 保存分析结果
        analysis_file = "PYTORCH_DETAILED_ANALYSIS.md"
        with open(analysis_file, 'w', encoding='utf-8') as f:
            f.write("# PyTorch版本详细分析结果\n\n")
            f.write(f"## 模型参数统计\n")
            f.write(f"- 总参数: {pytorch_info['total_params']:,} ({pytorch_info['total_params']/1e6:.2f}M)\n")
            f.write(f"- 可训练参数: {pytorch_info['trainable_params']:,}\n")
            f.write(f"- Backbone参数: {pytorch_info['backbone_params']:,}\n")
            f.write(f"- Neck参数: {pytorch_info['neck_params']:,}\n")
            f.write(f"- Head参数: {pytorch_info['head_params']:,}\n\n")
            
            f.write(f"## 特征图尺寸\n")
            for name, shape in pytorch_info['feature_shapes'].items():
                f.write(f"- {name}: {shape}\n")
            
            f.write(f"\n## 输出信息\n")
            f.write(f"- 输出类型: {type(pytorch_info['outputs'])}\n")
            if isinstance(pytorch_info['outputs'], (list, tuple)):
                for i, output in enumerate(pytorch_info['outputs']):
                    if hasattr(output, 'shape'):
                        f.write(f"- 输出{i}: {output.shape}\n")
        
        print(f"   分析结果已保存到: {analysis_file}")
    else:
        print(f"\n❌ PyTorch版本分析失败!")

if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
调试模型前向传播问题
分析为什么所有输出都相同
"""

import os
import sys
import cv2
import numpy as np
import jittor as jt
from pathlib import Path

# 添加路径
sys.path.append('.')
sys.path.append('./yolov6')

from models.perfect_gold_yolo import create_perfect_gold_yolo_model
from yolov6.data.data_augment import letterbox

def debug_model_forward():
    """调试模型前向传播"""
    print(f"🔍 调试模型前向传播问题")
    print("=" * 80)
    
    # 准备数据
    img_path = "/home/kyc/project/GOLD-YOLO/2008_001420.jpg"
    original_img = cv2.imread(img_path)
    img = letterbox(original_img, new_shape=500, stride=32, auto=False)[0]
    img_tensor_input = img.transpose((2, 0, 1))[::-1]
    img_tensor_input = np.ascontiguousarray(img_tensor_input)
    img_tensor_input = img_tensor_input.astype(np.float32) / 255.0
    img_tensor = jt.array(img_tensor_input).unsqueeze(0)
    
    print(f"📊 输入数据:")
    print(f"   图像张量: {img_tensor.shape}")
    print(f"   像素范围: [{float(img_tensor.min()):.6f}, {float(img_tensor.max()):.6f}]")
    print(f"   像素均值: {float(img_tensor.mean()):.6f}")
    print(f"   像素标准差: {float(img_tensor.std()):.6f}")
    
    # 创建模型
    print(f"\n🎯 创建模型:")
    model = create_perfect_gold_yolo_model()
    model.train()
    
    # 检查模型参数初始化
    print(f"\n🔍 检查模型参数初始化:")
    
    param_stats = {}
    for name, param in model.named_parameters():
        if param.requires_grad:
            param_mean = float(param.mean())
            param_std = float(param.std())
            param_min = float(param.min())
            param_max = float(param.max())
            
            param_stats[name] = {
                'mean': param_mean,
                'std': param_std,
                'min': param_min,
                'max': param_max,
                'shape': param.shape
            }
            
            # 只显示关键层
            if any(keyword in name for keyword in ['cls_pred', 'reg_pred', 'stem', 'head']):
                print(f"   {name}: 均值={param_mean:.6f}, 标准差={param_std:.6f}, 范围=[{param_min:.6f}, {param_max:.6f}]")
    
    # 检查是否有参数为0或常数
    zero_params = []
    constant_params = []
    
    for name, stats in param_stats.items():
        if abs(stats['std']) < 1e-8:
            if abs(stats['mean']) < 1e-8:
                zero_params.append(name)
            else:
                constant_params.append(name)
    
    if zero_params:
        print(f"\n⚠️ 零参数 ({len(zero_params)}个):")
        for name in zero_params[:5]:  # 只显示前5个
            print(f"     {name}")
        if len(zero_params) > 5:
            print(f"     ... 还有 {len(zero_params) - 5} 个")
    
    if constant_params:
        print(f"\n⚠️ 常数参数 ({len(constant_params)}个):")
        for name in constant_params[:5]:
            print(f"     {name}")
        if len(constant_params) > 5:
            print(f"     ... 还有 {len(constant_params) - 5} 个")
    
    # 逐层前向传播分析
    print(f"\n🔄 逐层前向传播分析:")
    
    # Hook函数来捕获中间输出
    layer_outputs = {}
    
    def create_hook(layer_name):
        def hook_fn(module, input, output):
            if isinstance(output, jt.Var):
                layer_outputs[layer_name] = {
                    'shape': output.shape,
                    'mean': float(output.mean()),
                    'std': float(output.std()),
                    'min': float(output.min()),
                    'max': float(output.max())
                }
            elif isinstance(output, (list, tuple)):
                for i, out in enumerate(output):
                    if isinstance(out, jt.Var):
                        layer_outputs[f"{layer_name}_out{i}"] = {
                            'shape': out.shape,
                            'mean': float(out.mean()),
                            'std': float(out.std()),
                            'min': float(out.min()),
                            'max': float(out.max())
                        }
        return hook_fn
    
    # 注册关键层的hook
    hooks = []
    
    # Backbone hooks
    if hasattr(model, 'backbone'):
        hook = model.backbone.register_forward_hook(create_hook('backbone'))
        hooks.append(hook)
    
    # Neck hooks
    if hasattr(model, 'neck'):
        hook = model.neck.register_forward_hook(create_hook('neck'))
        hooks.append(hook)
    
    # Head hooks
    if hasattr(model, 'head'):
        hook = model.head.register_forward_hook(create_hook('head'))
        hooks.append(hook)
        
        # Head子模块hooks
        if hasattr(model.head, 'cls_pred'):
            for i, cls_pred in enumerate(model.head.cls_pred):
                hook = cls_pred.register_forward_hook(create_hook(f'cls_pred_{i}'))
                hooks.append(hook)
        
        if hasattr(model.head, 'reg_pred'):
            for i, reg_pred in enumerate(model.head.reg_pred):
                hook = reg_pred.register_forward_hook(create_hook(f'reg_pred_{i}'))
                hooks.append(hook)
    
    # 前向传播
    print(f"   执行前向传播...")
    outputs = model(img_tensor)
    
    # 移除hooks
    for hook in hooks:
        hook.remove()
    
    # 分析中间层输出
    print(f"\n📊 中间层输出分析:")
    for layer_name, stats in layer_outputs.items():
        print(f"   {layer_name}: {stats['shape']}, 均值={stats['mean']:.6f}, 标准差={stats['std']:.6f}, 范围=[{stats['min']:.6f}, {stats['max']:.6f}]")
        
        # 检查异常输出
        if abs(stats['std']) < 1e-8:
            print(f"     ⚠️ 输出为常数！")
        if stats['min'] == stats['max']:
            print(f"     ⚠️ 所有值相同！")
    
    # 分析最终输出
    print(f"\n🎯 最终输出分析:")
    if isinstance(outputs, (list, tuple)):
        for i, output in enumerate(outputs):
            if isinstance(output, jt.Var):
                print(f"   输出{i}: {output.shape}")
                print(f"     均值: {float(output.mean()):.6f}")
                print(f"     标准差: {float(output.std()):.6f}")
                print(f"     范围: [{float(output.min()):.6f}, {float(output.max()):.6f}]")
                
                # 检查是否所有值相同
                if float(output.std()) < 1e-8:
                    print(f"     ❌ 所有值相同！这是问题所在！")
                    
                    # 检查具体值
                    unique_values = jt.unique(output.flatten())
                    print(f"     唯一值数量: {len(unique_values)}")
                    if len(unique_values) <= 5:
                        print(f"     唯一值: {[float(v) for v in unique_values]}")
    
    # 检查梯度计算
    print(f"\n🔍 检查梯度计算:")
    
    # 创建一个简单的损失
    if isinstance(outputs, (list, tuple)) and len(outputs) >= 2:
        pred_scores = outputs[1]
        pred_distri = outputs[2]
        
        # 简单的L2损失
        target_scores = jt.ones_like(pred_scores) * 0.5
        target_distri = jt.ones_like(pred_distri) * 2.0
        
        loss = ((pred_scores - target_scores) ** 2).mean() + ((pred_distri - target_distri) ** 2).mean()
        print(f"   简单损失: {float(loss):.6f}")
        
        # 反向传播
        optimizer = jt.optim.SGD(model.parameters(), lr=0.01)
        optimizer.zero_grad()
        optimizer.backward(loss)
        
        # 检查梯度
        grad_stats = {}
        for name, param in model.named_parameters():
            if param.requires_grad:
                try:
                    grad = param.opt_grad(optimizer)
                    if grad is not None:
                        grad_norm = float(grad.norm())
                        grad_stats[name] = grad_norm
                    else:
                        grad_stats[name] = 0.0
                except:
                    grad_stats[name] = 0.0
        
        # 统计梯度
        non_zero_grads = sum(1 for g in grad_stats.values() if g > 1e-8)
        total_grads = len(grad_stats)
        
        print(f"   有梯度的参数: {non_zero_grads}/{total_grads}")
        
        # 显示关键层梯度
        for name, grad_norm in grad_stats.items():
            if any(keyword in name for keyword in ['cls_pred', 'reg_pred', 'stem']) and grad_norm > 1e-8:
                print(f"     {name}: 梯度范数={grad_norm:.6f}")
    
    return {
        'layer_outputs': layer_outputs,
        'param_stats': param_stats,
        'outputs': outputs
    }

def main():
    print("🔍 模型前向传播调试")
    print("=" * 80)
    
    try:
        result = debug_model_forward()
        
        if result:
            print(f"\n✅ 调试完成!")
            
            # 总结问题
            outputs = result['outputs']
            if isinstance(outputs, (list, tuple)) and len(outputs) >= 2:
                pred_scores = outputs[1]
                pred_distri = outputs[2]
                
                scores_std = float(pred_scores.std())
                distri_std = float(pred_distri.std())
                
                if scores_std < 1e-8 and distri_std < 1e-8:
                    print(f"\n❌ 发现问题: 模型输出为常数")
                    print(f"   可能原因:")
                    print(f"   1. 模型初始化有问题")
                    print(f"   2. 某些层没有正确工作")
                    print(f"   3. 激活函数有问题")
                    print(f"   4. BatchNorm层有问题")
                else:
                    print(f"\n✅ 模型输出正常")
        else:
            print(f"\n❌ 调试失败!")
            
    except Exception as e:
        print(f"\n❌ 调试异常: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()

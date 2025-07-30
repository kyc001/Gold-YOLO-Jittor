#!/usr/bin/env python3
"""
深入调试梯度流问题
分析为什么大量BatchNorm层没有梯度
"""

import os
import sys
import cv2
import numpy as np
import jittor as jt
from pathlib import Path
import time

# 添加路径
sys.path.append('.')
sys.path.append('./yolov6')

from models.perfect_gold_yolo import create_perfect_gold_yolo_model
from yolov6.data.data_augment import letterbox
from yolov6.models.pytorch_aligned_losses import ComputeLoss  # 使用100%对齐PyTorch版本
from yolov6.utils.nms import non_max_suppression

def pytorch_exact_initialization(model):
    """完全照抄PyTorch版本的初始化"""
    for module in model.modules():
        if hasattr(module, 'initialize_biases'):
            module.initialize_biases()
            break
    return model

def debug_gradient_flow():
    """深入调试梯度流问题"""
    print(f"🔍 深入调试梯度流问题")
    print("=" * 80)
    
    # 准备数据
    label_file = "/home/kyc/project/GOLD-YOLO/2008_001420.txt"
    img_path = "/home/kyc/project/GOLD-YOLO/2008_001420.jpg"
    
    # 读取数据
    annotations = []
    with open(label_file, 'r') as f:
        for line in f:
            line = line.strip()
            if line:
                parts = line.split()
                if len(parts) >= 5:
                    cls_id = int(parts[0])
                    x_center = float(parts[1])
                    y_center = float(parts[2])
                    width = float(parts[3])
                    height = float(parts[4])
                    annotations.append([cls_id, x_center, y_center, width, height])
    
    original_img = cv2.imread(img_path)
    img = letterbox(original_img, new_shape=500, stride=32, auto=False)[0]
    img_tensor_input = img.transpose((2, 0, 1))[::-1]
    img_tensor_input = np.ascontiguousarray(img_tensor_input)
    img_tensor_input = img_tensor_input.astype(np.float32) / 255.0
    img_tensor = jt.array(img_tensor_input).unsqueeze(0)
    
    targets = []
    for ann in annotations:
        cls_id, x_center, y_center, width, height = ann
        targets.append([0, cls_id, x_center, y_center, width, height])
    targets_tensor = jt.array(targets, dtype=jt.float32).unsqueeze(0)
    
    print(f"📊 数据准备完成:")
    print(f"   图像张量: {img_tensor.shape}")
    print(f"   标签张量: {targets_tensor.shape}")
    
    # 创建模型
    print(f"\n🎯 创建模型并分析结构:")
    model = create_perfect_gold_yolo_model()
    model = pytorch_exact_initialization(model)
    
    # 创建100%对齐PyTorch版本的损失函数
    loss_fn = ComputeLoss(
        num_classes=20,
        ori_img_size=500,
        warmup_epoch=4,
        use_dfl=True,   # 使用100%对齐PyTorch版本
        reg_max=16,     # 使用100%对齐PyTorch版本
        fpn_strides=[8, 16, 32],
        grid_cell_size=5.0,
        grid_cell_offset=0.5,
        loss_weight={'class': 1.0, 'iou': 2.5, 'dfl': 0.5}
    )
    
    # 创建优化器
    optimizer = jt.optim.SGD(model.parameters(), lr=0.02, momentum=0.937, weight_decay=0.0005)
    
    print(f"\n🔍 分析模型参数和梯度:")
    
    # 统计所有参数
    total_params = 0
    params_with_grad = 0
    params_without_grad = 0
    
    print(f"\n📋 参数详细分析:")
    for name, param in model.named_parameters():
        total_params += 1
        if param.requires_grad:
            params_with_grad += 1
            grad_status = "✅ 需要梯度"
        else:
            params_without_grad += 1
            grad_status = "❌ 不需要梯度"
        
        # 只显示关键层
        if 'Inject' in name or 'embedding' in name or 'bn' in name:
            print(f"   {name}: {param.shape} - {grad_status}")
    
    print(f"\n📊 参数统计:")
    print(f"   总参数数: {total_params}")
    print(f"   需要梯度: {params_with_grad}")
    print(f"   不需要梯度: {params_without_grad}")
    
    # 前向传播
    print(f"\n🔄 执行前向传播:")
    model.train()
    outputs = model(img_tensor)
    
    print(f"   模型输出:")
    if isinstance(outputs, (list, tuple)):
        for i, output in enumerate(outputs):
            if hasattr(output, 'shape'):
                print(f"     输出{i}: {output.shape}")
            else:
                print(f"     输出{i}: {type(output)} (可能是list)")
    else:
        print(f"     输出: {outputs.shape if hasattr(outputs, 'shape') else type(outputs)}")
    
    # 计算损失
    print(f"\n💰 计算损失:")
    try:
        loss, loss_items = loss_fn(outputs, targets_tensor, epoch_num=1, step_num=1)
        print(f"   损失值: {float(loss.data.item()):.6f}")
        print(f"   损失项: {[float(item.data.item()) for item in loss_items]}")
    except Exception as e:
        print(f"   ❌ 损失计算失败: {e}")
        return
    
    # 反向传播
    print(f"\n⬅️ 执行反向传播:")
    optimizer.zero_grad()
    
    # 手动设置梯度计算
    jt.flags.use_cuda = 1
    
    try:
        optimizer.backward(loss)
        print(f"   ✅ 反向传播成功")
    except Exception as e:
        print(f"   ❌ 反向传播失败: {e}")
        return
    
    # 分析梯度
    print(f"\n🔍 分析梯度分布:")
    
    params_with_actual_grad = 0
    params_without_actual_grad = 0
    zero_grad_params = []
    
    for name, param in model.named_parameters():
        if param.requires_grad:
            try:
                # 使用Jittor正确的梯度API
                grad = param.opt_grad(optimizer)
                if grad is not None:
                    grad_norm = float(grad.norm().data.item())
                    if grad_norm > 1e-8:
                        params_with_actual_grad += 1
                        grad_status = f"✅ 梯度范数: {grad_norm:.6f}"
                    else:
                        params_without_actual_grad += 1
                        grad_status = f"⚠️ 梯度为零"
                        zero_grad_params.append(name)
                else:
                    params_without_actual_grad += 1
                    grad_status = f"❌ 无梯度"
                    zero_grad_params.append(name)
            except Exception as e:
                params_without_actual_grad += 1
                grad_status = f"❌ 梯度获取失败: {e}"
                zero_grad_params.append(name)
        else:
            grad_status = f"➖ 不需要梯度"
        
        # 只显示关键层
        if 'Inject' in name or 'embedding' in name or 'bn' in name:
            print(f"   {name}: {grad_status}")
    
    print(f"\n📊 梯度统计:")
    print(f"   有实际梯度: {params_with_actual_grad}")
    print(f"   梯度为零/无梯度: {params_without_actual_grad}")
    
    if zero_grad_params:
        print(f"\n⚠️ 梯度为零的参数 (前20个):")
        for name in zero_grad_params[:20]:
            print(f"     {name}")
        if len(zero_grad_params) > 20:
            print(f"     ... 还有 {len(zero_grad_params) - 20} 个")
    
    # 分析模型结构问题
    print(f"\n🏗️ 分析模型结构问题:")
    
    # 检查Inject模块是否被正确使用
    inject_modules = []
    for name, module in model.named_modules():
        if 'Inject' in name:
            inject_modules.append(name)
    
    print(f"   发现Inject模块: {len(inject_modules)}个")
    for name in inject_modules:
        print(f"     {name}")
    
    # 检查模型前向传播路径
    print(f"\n🛤️ 检查前向传播路径:")
    
    # 使用hook来跟踪哪些模块被调用
    called_modules = []
    
    def forward_hook(module, input, output):
        called_modules.append(module.__class__.__name__)
    
    # 注册hook
    hooks = []
    for module in model.modules():
        if 'Inject' in module.__class__.__name__ or 'embedding' in str(module).lower():
            hook = module.register_forward_hook(forward_hook)
            hooks.append(hook)
    
    # 重新前向传播
    called_modules.clear()
    with jt.no_grad():
        _ = model(img_tensor)
    
    # 移除hook
    for hook in hooks:
        hook.remove()
    
    print(f"   被调用的关键模块: {set(called_modules)}")
    
    # 建议修复方案
    print(f"\n🔧 问题分析和修复建议:")
    
    if params_without_actual_grad > params_with_actual_grad:
        print(f"   ❌ 大量参数没有梯度，可能原因:")
        print(f"     1. 某些模块没有被正确连接到计算图")
        print(f"     2. 损失函数没有正确反向传播到所有参数")
        print(f"     3. 模型结构中存在断开的分支")
        print(f"     4. BatchNorm层在eval模式下运行")
    
    if 'Inject' not in str(called_modules):
        print(f"   ❌ Inject模块可能没有被正确调用")
        print(f"     建议检查neck部分的前向传播逻辑")
    
    print(f"\n💡 修复建议:")
    print(f"   1. 确保模型在train模式下运行")
    print(f"   2. 检查neck部分的Inject模块连接")
    print(f"   3. 使用更简单的损失函数进行测试")
    print(f"   4. 检查模型初始化是否正确")
    
    return {
        'total_params': total_params,
        'params_with_grad': params_with_grad,
        'params_without_grad': params_without_grad,
        'params_with_actual_grad': params_with_actual_grad,
        'params_without_actual_grad': params_without_actual_grad,
        'zero_grad_params': zero_grad_params,
        'inject_modules': inject_modules,
        'called_modules': called_modules
    }

def main():
    print("🔍 深入调试梯度流问题")
    print("=" * 80)
    
    result = debug_gradient_flow()
    
    if result:
        print(f"\n📊 梯度流调试完成!")
        print(f"   参数总数: {result['total_params']}")
        print(f"   有实际梯度: {result['params_with_actual_grad']}")
        print(f"   无实际梯度: {result['params_without_actual_grad']}")
        print(f"   Inject模块数: {len(result['inject_modules'])}")
        
        if result['params_without_actual_grad'] > result['params_with_actual_grad']:
            print(f"\n❌ 梯度流存在严重问题！")
            print(f"   需要立即修复模型结构或损失函数")
        else:
            print(f"\n✅ 梯度流基本正常")
    else:
        print(f"\n❌ 梯度流调试失败!")

if __name__ == "__main__":
    main()

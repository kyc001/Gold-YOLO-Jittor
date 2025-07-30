#!/usr/bin/env python3
"""
全面问题扫描
深入找出所有问题，不放过任何细节
"""

import os
import sys
import cv2
import numpy as np
import jittor as jt
from pathlib import Path
import math

# 添加路径
sys.path.append('.')
sys.path.append('./yolov6')

from models.perfect_gold_yolo import create_perfect_gold_yolo_model
from yolov6.data.data_augment import letterbox
from yolov6.models.pytorch_aligned_losses import ComputeLoss

def scan_model_initialization():
    """扫描模型初始化问题"""
    print(f"🔍 扫描1: 模型初始化问题")
    print("-" * 60)
    
    model = create_perfect_gold_yolo_model()
    
    # 检查所有参数的初始化
    init_problems = []
    
    for name, param in model.named_parameters():
        if param.requires_grad:
            param_mean = float(param.mean())
            param_std = float(param.std())
            param_min = float(param.min())
            param_max = float(param.max())
            
            # 检查异常初始化
            if abs(param_std) < 1e-8:
                if abs(param_mean) < 1e-8:
                    init_problems.append(f"❌ {name}: 全零参数")
                else:
                    init_problems.append(f"⚠️ {name}: 常数参数 (值={param_mean:.6f})")
            
            # 检查过大的初始化
            if param_std > 10.0:
                init_problems.append(f"⚠️ {name}: 初始化过大 (std={param_std:.6f})")
            
            # 检查NaN或Inf
            if not jt.isfinite(param).all():
                init_problems.append(f"❌ {name}: 包含NaN或Inf")
    
    print(f"   检查了 {len(list(model.named_parameters()))} 个参数")
    if init_problems:
        print(f"   发现 {len(init_problems)} 个初始化问题:")
        for problem in init_problems[:10]:  # 只显示前10个
            print(f"     {problem}")
        if len(init_problems) > 10:
            print(f"     ... 还有 {len(init_problems) - 10} 个问题")
    else:
        print(f"   ✅ 所有参数初始化正常")
    
    return init_problems

def scan_forward_propagation():
    """扫描前向传播问题"""
    print(f"\n🔍 扫描2: 前向传播问题")
    print("-" * 60)
    
    # 准备测试数据
    img_path = "/home/kyc/project/GOLD-YOLO/2008_001420.jpg"
    original_img = cv2.imread(img_path)
    img = letterbox(original_img, new_shape=500, stride=32, auto=False)[0]
    img_tensor_input = img.transpose((2, 0, 1))[::-1]
    img_tensor_input = np.ascontiguousarray(img_tensor_input)
    img_tensor_input = img_tensor_input.astype(np.float32) / 255.0
    img_tensor = jt.array(img_tensor_input).unsqueeze(0)
    
    model = create_perfect_gold_yolo_model()
    model.train()
    
    forward_problems = []
    
    # 检查前向传播
    try:
        outputs = model(img_tensor)
        
        # 检查输出格式
        if not isinstance(outputs, (list, tuple)):
            forward_problems.append("❌ 输出不是tuple/list格式")
        elif len(outputs) != 3:
            forward_problems.append(f"❌ 输出数量错误: 期望3个，实际{len(outputs)}个")
        else:
            feats, pred_scores, pred_distri = outputs
            
            # 检查feats
            if not isinstance(feats, list):
                forward_problems.append("❌ feats不是list格式")
            elif len(feats) != 3:
                forward_problems.append(f"❌ feats数量错误: 期望3个，实际{len(feats)}个")
            
            # 检查pred_scores
            if not hasattr(pred_scores, 'shape'):
                forward_problems.append("❌ pred_scores没有shape属性")
            else:
                if pred_scores.shape != (1, 5249, 20):
                    forward_problems.append(f"❌ pred_scores形状错误: 期望[1,5249,20]，实际{pred_scores.shape}")
                
                # 检查数值范围
                scores_min = float(pred_scores.min())
                scores_max = float(pred_scores.max())
                scores_std = float(pred_scores.std())
                
                if scores_std < 1e-8:
                    forward_problems.append(f"❌ pred_scores所有值相同: {scores_min:.6f}")
                elif scores_min == scores_max:
                    forward_problems.append(f"❌ pred_scores无变化: [{scores_min:.6f}, {scores_max:.6f}]")
                elif not (0.0 <= scores_min <= 1.0 and 0.0 <= scores_max <= 1.0):
                    forward_problems.append(f"⚠️ pred_scores范围异常: [{scores_min:.6f}, {scores_max:.6f}]")
            
            # 检查pred_distri
            if not hasattr(pred_distri, 'shape'):
                forward_problems.append("❌ pred_distri没有shape属性")
            else:
                if pred_distri.shape != (1, 5249, 4):
                    forward_problems.append(f"❌ pred_distri形状错误: 期望[1,5249,4]，实际{pred_distri.shape}")
                
                # 检查数值范围
                distri_min = float(pred_distri.min())
                distri_max = float(pred_distri.max())
                distri_std = float(pred_distri.std())
                
                if distri_std < 1e-8:
                    forward_problems.append(f"❌ pred_distri所有值相同: {distri_min:.6f}")
                elif distri_min == distri_max:
                    forward_problems.append(f"❌ pred_distri无变化: [{distri_min:.6f}, {distri_max:.6f}]")
                elif distri_min < 0:
                    forward_problems.append(f"⚠️ pred_distri有负值: [{distri_min:.6f}, {distri_max:.6f}]")
    
    except Exception as e:
        forward_problems.append(f"❌ 前向传播异常: {e}")
    
    if forward_problems:
        print(f"   发现 {len(forward_problems)} 个前向传播问题:")
        for problem in forward_problems:
            print(f"     {problem}")
    else:
        print(f"   ✅ 前向传播正常")
    
    return forward_problems

def scan_loss_function():
    """扫描损失函数问题"""
    print(f"\n🔍 扫描3: 损失函数问题")
    print("-" * 60)
    
    # 准备数据
    label_file = "/home/kyc/project/GOLD-YOLO/2008_001420.txt"
    img_path = "/home/kyc/project/GOLD-YOLO/2008_001420.jpg"
    
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
    
    model = create_perfect_gold_yolo_model()
    model.train()
    
    loss_problems = []
    
    try:
        # 创建损失函数
        loss_fn = ComputeLoss(
            num_classes=20,
            ori_img_size=500,
            warmup_epoch=0,
            use_dfl=False,
            reg_max=0,
            fpn_strides=[8, 16, 32],
            grid_cell_size=5.0,
            grid_cell_offset=0.5,
            loss_weight={'class': 1.0, 'iou': 2.5, 'dfl': 0.5}
        )
        
        # 前向传播
        outputs = model(img_tensor)
        
        # 计算损失
        loss, loss_items = loss_fn(outputs, targets_tensor, epoch_num=0, step_num=0)
        
        # 检查损失值
        loss_value = float(loss.data.item())
        loss_items_values = [float(item.data.item()) for item in loss_items]
        
        print(f"   损失值: {loss_value:.6f}")
        print(f"   损失项: {loss_items_values}")
        
        # 检查损失异常
        if not math.isfinite(loss_value):
            loss_problems.append(f"❌ 总损失为NaN或Inf: {loss_value}")
        elif loss_value > 1000:
            loss_problems.append(f"⚠️ 总损失过大: {loss_value:.6f}")
        elif loss_value < 0:
            loss_problems.append(f"❌ 总损失为负: {loss_value:.6f}")
        
        # 检查各项损失
        loss_names = ['IoU Loss', 'DFL Loss', 'Class Loss']
        for i, (name, value) in enumerate(zip(loss_names, loss_items_values)):
            if not math.isfinite(value):
                loss_problems.append(f"❌ {name}为NaN或Inf: {value}")
            elif value > 10000:
                loss_problems.append(f"⚠️ {name}过大: {value:.6f}")
            elif value < 0:
                loss_problems.append(f"❌ {name}为负: {value:.6f}")
        
        # 检查分类损失为0的问题
        if loss_items_values[2] == 0.0:
            loss_problems.append(f"⚠️ 分类损失为0，可能没有正样本")
        
        # 测试梯度计算
        optimizer = jt.optim.SGD(model.parameters(), lr=0.01)
        optimizer.zero_grad()
        
        try:
            optimizer.backward(loss)
            
            # 检查梯度
            grad_count = 0
            nan_grad_count = 0
            large_grad_count = 0
            
            for name, param in model.named_parameters():
                if param.requires_grad:
                    try:
                        grad = param.opt_grad(optimizer)
                        if grad is not None:
                            grad_norm = float(grad.norm())
                            if not math.isfinite(grad_norm):
                                nan_grad_count += 1
                            elif grad_norm > 100:
                                large_grad_count += 1
                            grad_count += 1
                    except:
                        pass
            
            print(f"   梯度统计: {grad_count}个有梯度, {nan_grad_count}个NaN梯度, {large_grad_count}个大梯度")
            
            if nan_grad_count > 0:
                loss_problems.append(f"❌ {nan_grad_count}个参数梯度为NaN")
            if large_grad_count > 10:
                loss_problems.append(f"⚠️ {large_grad_count}个参数梯度过大(>100)")
                
        except Exception as e:
            loss_problems.append(f"❌ 梯度计算失败: {e}")
    
    except Exception as e:
        loss_problems.append(f"❌ 损失计算失败: {e}")
    
    if loss_problems:
        print(f"   发现 {len(loss_problems)} 个损失函数问题:")
        for problem in loss_problems:
            print(f"     {problem}")
    else:
        print(f"   ✅ 损失函数正常")
    
    return loss_problems

def scan_numerical_stability():
    """扫描数值稳定性问题"""
    print(f"\n🔍 扫描4: 数值稳定性问题")
    print("-" * 60)
    
    stability_problems = []
    
    # 测试不同学习率的稳定性
    learning_rates = [0.001, 0.01, 0.1, 1.0]
    
    for lr in learning_rates:
        try:
            # 简单测试
            model = create_perfect_gold_yolo_model()
            optimizer = jt.optim.SGD(model.parameters(), lr=lr)
            
            # 创建简单损失
            dummy_loss = jt.sum(jt.stack([p.sum() for p in model.parameters() if p.requires_grad]))
            
            optimizer.zero_grad()
            optimizer.backward(dummy_loss)
            
            # 检查梯度范数
            total_grad_norm = 0.0
            for param in model.parameters():
                if param.requires_grad:
                    try:
                        grad = param.opt_grad(optimizer)
                        if grad is not None:
                            total_grad_norm += float(grad.norm()) ** 2
                    except:
                        pass
            
            total_grad_norm = math.sqrt(total_grad_norm)
            
            print(f"   学习率 {lr}: 总梯度范数 {total_grad_norm:.6f}")
            
            if not math.isfinite(total_grad_norm):
                stability_problems.append(f"❌ 学习率{lr}: 梯度为NaN或Inf")
            elif total_grad_norm > 1000:
                stability_problems.append(f"⚠️ 学习率{lr}: 梯度过大 ({total_grad_norm:.2f})")
                
        except Exception as e:
            stability_problems.append(f"❌ 学习率{lr}测试失败: {e}")
    
    if stability_problems:
        print(f"   发现 {len(stability_problems)} 个数值稳定性问题:")
        for problem in stability_problems:
            print(f"     {problem}")
    else:
        print(f"   ✅ 数值稳定性正常")
    
    return stability_problems

def main():
    print("🔍 GOLD-YOLO 全面问题扫描")
    print("=" * 80)
    print("深入找出所有问题，不放过任何细节")
    print("=" * 80)
    
    all_problems = []
    
    # 执行所有扫描
    all_problems.extend(scan_model_initialization())
    all_problems.extend(scan_forward_propagation())
    all_problems.extend(scan_loss_function())
    all_problems.extend(scan_numerical_stability())
    
    # 总结
    print(f"\n📊 扫描总结")
    print("=" * 80)
    print(f"总共发现 {len(all_problems)} 个问题:")
    
    if all_problems:
        for i, problem in enumerate(all_problems, 1):
            print(f"{i:2d}. {problem}")
        
        print(f"\n🔧 需要修复的关键问题:")
        critical_problems = [p for p in all_problems if p.startswith("❌")]
        warning_problems = [p for p in all_problems if p.startswith("⚠️")]
        
        print(f"   严重问题: {len(critical_problems)}个")
        print(f"   警告问题: {len(warning_problems)}个")
        
        if critical_problems:
            print(f"\n   严重问题列表:")
            for problem in critical_problems:
                print(f"     {problem}")
    else:
        print("✅ 没有发现问题，模型状态良好")
    
    return all_problems

if __name__ == "__main__":
    main()

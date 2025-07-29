#!/usr/bin/env python3
"""
调试分类学习失败问题
分析为什么期望类别分数全为0，aeroplane分数为1
"""

import os
import sys
import cv2
import numpy as np
import jittor as jt
import time
from pathlib import Path

# 添加路径
sys.path.append('.')
sys.path.append('./yolov6')

from models.perfect_gold_yolo import create_perfect_gold_yolo_model
from yolov6.data.data_augment import letterbox
from yolov6.models.losses import ComputeLoss

# VOC数据集类别名称
VOC_CLASSES = [
    'aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car', 'cat', 'chair', 'cow',
    'diningtable', 'dog', 'horse', 'motorbike', 'person', 'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor'
]

def pytorch_exact_initialization(model):
    """完全照抄PyTorch版本的初始化"""
    for name, module in model.named_modules():
        if hasattr(module, 'initialize_biases'):
            module.initialize_biases()
            break
    return model

def debug_classification_failure():
    """调试分类学习失败问题"""
    print(f"🔧 调试分类学习失败问题")
    print("=" * 50)
    
    # 准备数据
    label_file = "/home/kyc/project/GOLD-YOLO/2008_001420.txt"
    img_path = "/home/kyc/project/GOLD-YOLO/2008_001420.jpg"
    
    # 读取真实标注
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
    
    target_counts = {}
    for ann in annotations:
        cls_name = VOC_CLASSES[ann[0]]
        target_counts[cls_name] = target_counts.get(cls_name, 0) + 1
    
    print(f"📋 期望检测结果: {target_counts}")
    print(f"   期望类别ID: {[ann[0] for ann in annotations]}")
    
    # 准备输入
    original_img = cv2.imread(img_path)
    img = letterbox(original_img, new_shape=640, stride=32, auto=False)[0]
    img = img.transpose((2, 0, 1))[::-1]
    img = np.ascontiguousarray(img)
    img = img.astype(np.float32) / 255.0
    img_tensor = jt.array(img).unsqueeze(0)
    
    # 准备标签
    targets = []
    for ann in annotations:
        cls_id, x_center, y_center, width, height = ann
        targets.append([0, cls_id, x_center, y_center, width, height])
    targets_tensor = jt.array(targets, dtype=jt.float32).unsqueeze(0)
    
    print(f"📊 标签分析:")
    print(f"   targets_tensor形状: {targets_tensor.shape}")
    targets_np = targets_tensor.numpy() if hasattr(targets_tensor, 'numpy') else targets_tensor
    print(f"   targets_tensor内容: {targets_np}")
    
    # 创建模型
    model = create_perfect_gold_yolo_model()
    model = pytorch_exact_initialization(model)
    model.train()
    
    # 创建损失函数和优化器
    loss_fn = ComputeLoss(
        num_classes=20,
        ori_img_size=640,
        warmup_epoch=4,
        use_dfl=False,
        reg_max=0,
        iou_type='giou',
        loss_weight={'class': 1.0, 'iou': 2.5, 'dfl': 0.5}
    )
    optimizer = jt.optim.AdamW(model.parameters(), lr=0.05)
    
    print(f"\n🔍 深入分析分类学习过程:")
    
    # 训练前检查
    print(f"\n📊 训练前模型输出:")
    model.eval()
    with jt.no_grad():
        initial_outputs = model(img_tensor)

        # 检查输出格式
        if isinstance(initial_outputs, tuple):
            print(f"   输出是tuple，长度: {len(initial_outputs)}")
            # 使用推理模式的输出格式
            if len(initial_outputs) >= 3:
                # 推理模式：(pred_scores, pred_distri, ...)
                pred_scores = initial_outputs[1]  # [1, 8400, 20]
                pred_distri = initial_outputs[2]  # [1, 8400, 4]

                print(f"   pred_scores形状: {pred_scores.shape}")
                print(f"   pred_distri形状: {pred_distri.shape}")
                print(f"   pred_scores范围: [{pred_scores.min():.6f}, {pred_scores.max():.6f}]")

                # 检查所有类别的初始分数
                print(f"   所有类别的初始分数:")
                for cls_id in range(20):
                    cls_scores = pred_scores[0, :, cls_id]
                    max_score = float(cls_scores.max())
                    if max_score > 0.005:  # 只显示有意义的分数
                        cls_name = VOC_CLASSES[cls_id]
                        print(f"     {cls_name}(类别{cls_id}): 最大{max_score:.6f}")
        else:
            # 训练模式的输出格式
            coords = initial_outputs[..., :4]
            objectness = initial_outputs[..., 4]
            classes = initial_outputs[..., 5:]

            print(f"   坐标范围: [{coords.min():.3f}, {coords.max():.3f}]")
            print(f"   objectness范围: [{objectness.min():.3f}, {objectness.max():.3f}]")
            print(f"   类别分数范围: [{classes.min():.6f}, {classes.max():.6f}]")

            # 检查所有类别的初始分数
            print(f"   所有类别的初始分数:")
            for cls_id in range(20):
                cls_scores = classes[0, :, cls_id]
                max_score = float(cls_scores.max())
                if max_score > 0.005:  # 只显示有意义的分数
                    cls_name = VOC_CLASSES[cls_id]
                    print(f"     {cls_name}(类别{cls_id}): 最大{max_score:.6f}")

    model.train()
    
    # 训练循环 - 详细分析
    for epoch in range(50):
        # 前向传播
        outputs = model(img_tensor)
        
        # 计算损失
        loss, loss_items = loss_fn(outputs, targets_tensor, epoch_num=epoch+1, step_num=1)
        
        # 优化
        optimizer.step(loss)
        
        epoch_loss = float(loss.numpy())
        
        # 每10轮详细分析
        if (epoch + 1) % 10 == 0:
            print(f"\n   Epoch {epoch+1}: Loss {epoch_loss:.6f}")
            
            # 分析模型输出
            model.eval()
            with jt.no_grad():
                test_outputs = model(img_tensor)
                coords = test_outputs[..., :4]
                objectness = test_outputs[..., 4]
                classes = test_outputs[..., 5:]
                
                print(f"     模型输出分析:")
                print(f"       坐标范围: [{coords.min():.3f}, {coords.max():.3f}]")
                print(f"       objectness范围: [{objectness.min():.3f}, {objectness.max():.3f}]")
                print(f"       类别分数范围: [{classes.min():.6f}, {classes.max():.6f}]")
                
                # 检查期望类别的分数变化
                expected_classes = [3, 11, 14]  # boat, dog, person
                print(f"     期望类别分数变化:")
                for cls_id in expected_classes:
                    cls_scores = classes[0, :, cls_id]
                    max_score = float(cls_scores.max())
                    mean_score = float(cls_scores.mean())
                    nonzero_count = int((cls_scores > 0.001).sum())
                    print(f"       {VOC_CLASSES[cls_id]}(类别{cls_id}): 最大{max_score:.6f}, 平均{mean_score:.6f}, 非零{nonzero_count}")
                
                # 检查aeroplane的分数
                aero_scores = classes[0, :, 0]
                aero_max_score = float(aero_scores.max())
                aero_mean_score = float(aero_scores.mean())
                aero_nonzero_count = int((aero_scores > 0.001).sum())
                print(f"       aeroplane(类别0): 最大{aero_max_score:.6f}, 平均{aero_mean_score:.6f}, 非零{aero_nonzero_count}")
                
                # 检查所有类别的分数分布
                print(f"     所有类别分数统计:")
                for cls_id in range(20):
                    cls_scores = classes[0, :, cls_id]
                    max_score = float(cls_scores.max())
                    if max_score > 0.01:  # 只显示有意义的分数
                        mean_score = float(cls_scores.mean())
                        nonzero_count = int((cls_scores > 0.001).sum())
                        cls_name = VOC_CLASSES[cls_id]
                        print(f"       {cls_name}(类别{cls_id}): 最大{max_score:.6f}, 平均{mean_score:.6f}, 非零{nonzero_count}")
                
                # **关键检查：分类头的权重和梯度**
                print(f"     分类头权重和梯度分析:")
                for name, param in model.named_parameters():
                    if 'cls_preds' in name and param.requires_grad:
                        if param.grad is not None:
                            grad_norm = float(param.grad.norm())
                            weight_norm = float(param.norm())
                            print(f"       {name}: 权重范数{weight_norm:.6f}, 梯度范数{grad_norm:.6f}")
                        else:
                            weight_norm = float(param.norm())
                            print(f"       {name}: 权重范数{weight_norm:.6f}, 梯度为None")
                
                # **关键检查：损失函数的分类损失**
                if hasattr(loss_items, '__len__') and len(loss_items) >= 3:
                    cls_loss = float(loss_items[0]) if len(loss_items) > 0 else 0.0
                    iou_loss = float(loss_items[1]) if len(loss_items) > 1 else 0.0
                    dfl_loss = float(loss_items[2]) if len(loss_items) > 2 else 0.0
                    print(f"     损失分解: 分类{cls_loss:.6f}, IoU{iou_loss:.6f}, DFL{dfl_loss:.6f}")
                    
                    if cls_loss < 0.001:
                        print(f"     ⚠️ 分类损失过小，可能学习率过低或损失计算有问题")
                
                # 检查是否有学习进展
                if epoch == 10:
                    # 保存第10轮的分数作为基准
                    baseline_scores = {}
                    for cls_id in expected_classes:
                        cls_scores = classes[0, :, cls_id]
                        baseline_scores[cls_id] = float(cls_scores.max())
                elif epoch == 40:
                    # 检查第40轮相比第10轮的进展
                    print(f"     学习进展分析 (Epoch 10 vs 40):")
                    for cls_id in expected_classes:
                        cls_scores = classes[0, :, cls_id]
                        current_score = float(cls_scores.max())
                        if cls_id in baseline_scores:
                            improvement = current_score - baseline_scores[cls_id]
                            cls_name = VOC_CLASSES[cls_id]
                            print(f"       {cls_name}(类别{cls_id}): {baseline_scores[cls_id]:.6f} -> {current_score:.6f} (改进{improvement:+.6f})")
                            
                            if improvement < 0.001:
                                print(f"       ❌ {cls_name}没有学习进展！")
                            else:
                                print(f"       ✅ {cls_name}有学习进展")
            
            model.train()
    
    print(f"\n📊 最终分析结果:")
    print(f"如果期望类别分数仍然为0，说明分类学习完全失败")
    print(f"可能的原因：")
    print(f"1. 标签分配问题 - TaskAlignedAssigner没有正确分配正样本")
    print(f"2. 损失函数问题 - VarifocalLoss计算错误")
    print(f"3. 分类头初始化问题 - 权重初始化不当")
    print(f"4. 学习率问题 - 学习率过低或过高")
    print(f"5. 梯度传播问题 - 分类头没有接收到梯度")
    
    return False

def main():
    print("🔥 调试分类学习失败问题")
    print("=" * 70)
    print("目标：找出为什么期望类别分数全为0")
    print("=" * 70)
    
    debug_classification_failure()

if __name__ == "__main__":
    main()

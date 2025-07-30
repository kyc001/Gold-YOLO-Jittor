#!/usr/bin/env python3
"""
深入对齐PyTorch版本
找到为什么参数一样但训练效果不同的根本原因
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
from yolov6.assigners.anchor_generator import generate_anchors

def debug_target_assignment():
    """深入调试目标分配过程"""
    print(f"🎯 深入调试目标分配过程")
    print("=" * 80)
    
    # 准备数据
    label_file = "/home/kyc/project/GOLD-YOLO/2008_001420.txt"
    img_path = "/home/kyc/project/GOLD-YOLO/2008_001420.jpg"
    
    # 读取标注
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
    
    print(f"📋 标注信息:")
    for i, ann in enumerate(annotations):
        cls_id, x_center, y_center, width, height = ann
        print(f"   目标{i+1}: 类别={cls_id}, 中心=({x_center:.3f},{y_center:.3f}), 尺寸=({width:.3f},{height:.3f})")
    
    # 读取图像
    original_img = cv2.imread(img_path)
    img = letterbox(original_img, new_shape=500, stride=32, auto=False)[0]
    img_tensor_input = img.transpose((2, 0, 1))[::-1]
    img_tensor_input = np.ascontiguousarray(img_tensor_input)
    img_tensor_input = img_tensor_input.astype(np.float32) / 255.0
    img_tensor = jt.array(img_tensor_input).unsqueeze(0)
    
    # 准备标签
    targets = []
    for ann in annotations:
        cls_id, x_center, y_center, width, height = ann
        targets.append([0, cls_id, x_center, y_center, width, height])
    targets_tensor = jt.array(targets, dtype=jt.float32).unsqueeze(0)
    
    print(f"\n📊 数据格式:")
    print(f"   图像张量: {img_tensor.shape}")
    print(f"   标签张量: {targets_tensor.shape}")
    
    # 创建模型和损失函数
    model = create_perfect_gold_yolo_model()
    model.train()
    
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
    feats, pred_scores, pred_distri = outputs
    
    print(f"\n🔄 模型输出:")
    print(f"   特征列表: {len(feats)}个")
    print(f"   预测分数: {pred_scores.shape}")
    print(f"   预测距离: {pred_distri.shape}")
    
    # 深入分析目标分配过程
    print(f"\n🎯 深入分析目标分配:")
    
    # 1. 生成anchor points
    anchors, anchor_points, num_anchors_list, stride_tensor = generate_anchors(feats, loss_fn.fpn_strides, loss_fn.grid_cell_size, loss_fn.grid_cell_offset)
    print(f"   Anchor点数量: {anchor_points.shape[0]}")
    print(f"   Anchor点形状: {anchor_points.shape}")
    print(f"   步长张量形状: {stride_tensor.shape}")
    
    # 检查anchor points的分布
    anchor_points_np = anchor_points.data
    print(f"   Anchor点范围: x[{anchor_points_np[:, 0].min():.1f}, {anchor_points_np[:, 0].max():.1f}], y[{anchor_points_np[:, 1].min():.1f}, {anchor_points_np[:, 1].max():.1f}]")
    
    # 2. 准备GT数据 (通过损失函数的预处理)
    batch_size = 1
    gt_bboxes_scale = jt.full((1, 4), 500, dtype=jt.float32)
    processed_targets = loss_fn.preprocess(targets_tensor, batch_size, gt_bboxes_scale)
    gt_labels = processed_targets[:, :, :1]
    gt_bboxes = processed_targets[:, :, 1:]
    
    print(f"\n📋 GT数据:")
    print(f"   GT标签形状: {gt_labels.shape}")
    print(f"   GT框形状: {gt_bboxes.shape}")
    print(f"   GT标签: {gt_labels.data}")
    print(f"   GT框: {gt_bboxes.data}")

    # GT框已经是像素坐标了（经过preprocess处理）
    gt_bboxes_scaled = gt_bboxes
    print(f"   GT框(像素): {gt_bboxes_scaled.data}")
    
    # 3. 解码预测框
    pred_bboxes = loss_fn.bbox_decode(anchor_points, pred_distri)
    print(f"\n📦 预测框:")
    print(f"   预测框形状: {pred_bboxes.shape}")
    print(f"   预测框范围: [{float(pred_bboxes.min()):.2f}, {float(pred_bboxes.max()):.2f}]")
    
    # 4. 执行目标分配
    print(f"\n🎯 执行目标分配:")
    try:
        # 使用detach避免梯度计算
        pred_scores_detached = pred_scores.detach()
        pred_bboxes_scaled = pred_bboxes.detach() * stride_tensor
        anchor_points_scaled = anchor_points * stride_tensor
        
        print(f"   输入到分配器:")
        print(f"     pred_scores: {pred_scores_detached.shape}")
        print(f"     pred_bboxes_scaled: {pred_bboxes_scaled.shape}")
        print(f"     anchor_points_scaled: {anchor_points_scaled.shape}")
        print(f"     gt_labels: {gt_labels.shape}")
        print(f"     gt_bboxes_scaled: {gt_bboxes_scaled.shape}")
        
        # 调用目标分配器 (使用formal_assigner，因为epoch_num=0 >= warmup_epoch=0)
        mask_gt = (gt_bboxes_scaled.sum(-1, keepdim=True) > 0).float()
        print(f"   mask_gt形状: {mask_gt.shape}")
        print(f"   mask_gt内容: {mask_gt.data}")

        assigned_labels, assigned_bboxes, assigned_scores, fg_mask = loss_fn.formal_assigner(
            pred_scores_detached, pred_bboxes_scaled, anchor_points_scaled,
            gt_labels, gt_bboxes_scaled, mask_gt
        )
        
        print(f"\n✅ 目标分配成功:")
        print(f"   分配标签形状: {assigned_labels.shape}")
        print(f"   分配框形状: {assigned_bboxes.shape}")
        print(f"   分配分数形状: {assigned_scores.shape}")
        print(f"   前景掩码形状: {fg_mask.shape}")
        
        # 分析分配结果
        assigned_labels_np = assigned_labels.data
        assigned_scores_np = assigned_scores.data
        
        # 统计正样本
        pos_mask = assigned_labels_np > 0
        num_pos = pos_mask.sum()
        
        print(f"\n📊 分配结果分析:")
        print(f"   总anchor数: {len(assigned_labels_np)}")
        print(f"   正样本数: {int(num_pos)}")
        print(f"   负样本数: {int(len(assigned_labels_np) - num_pos)}")
        print(f"   正样本比例: {float(num_pos) / len(assigned_labels_np) * 100:.2f}%")
        
        if num_pos > 0:
            pos_labels = assigned_labels_np[pos_mask]
            pos_scores = assigned_scores_np[pos_mask]
            
            print(f"   正样本标签: {pos_labels}")
            print(f"   正样本分数范围: [{pos_scores.min():.4f}, {pos_scores.max():.4f}]")
            print(f"   正样本分数均值: {pos_scores.mean():.4f}")
            
            # 按类别统计
            unique_labels = np.unique(pos_labels)
            for label in unique_labels:
                count = (pos_labels == label).sum()
                print(f"   类别{int(label)}的正样本数: {count}")
        else:
            print(f"   ❌ 没有正样本！这就是分类损失为0的原因！")
            
            # 分析为什么没有正样本
            print(f"\n🔍 分析为什么没有正样本:")
            
            # 检查GT框和anchor的重叠
            print(f"   GT框尺寸分析:")
            for i, gt_box in enumerate(gt_bboxes_scaled.data):
                x_center, y_center, width, height = gt_box
                x1 = x_center - width/2
                y1 = y_center - height/2
                x2 = x_center + width/2
                y2 = y_center + height/2
                area = width * height
                print(f"     GT{i+1}: 中心({x_center:.1f},{y_center:.1f}), 尺寸({width:.1f}x{height:.1f}), 面积{area:.1f}")
                print(f"            边界({x1:.1f},{y1:.1f}) -> ({x2:.1f},{y2:.1f})")
            
            # 检查anchor points是否覆盖GT区域
            print(f"   Anchor覆盖分析:")
            for i, gt_box in enumerate(gt_bboxes_scaled.data):
                x_center, y_center, width, height = gt_box
                
                # 找到最近的anchor points
                distances = ((anchor_points_scaled.data[:, 0] - x_center)**2 + 
                           (anchor_points_scaled.data[:, 1] - y_center)**2)**0.5
                min_dist_idx = distances.argmin()
                min_dist = distances[min_dist_idx]
                closest_anchor = anchor_points_scaled.data[min_dist_idx]
                
                print(f"     GT{i+1}最近anchor: 距离{min_dist:.1f}, 位置({closest_anchor[0]:.1f},{closest_anchor[1]:.1f})")
                
                # 检查在GT框内的anchor数量
                x1, y1 = x_center - width/2, y_center - height/2
                x2, y2 = x_center + width/2, y_center + height/2
                
                inside_mask = ((anchor_points_scaled.data[:, 0] >= x1) & 
                              (anchor_points_scaled.data[:, 0] <= x2) &
                              (anchor_points_scaled.data[:, 1] >= y1) & 
                              (anchor_points_scaled.data[:, 1] <= y2))
                inside_count = inside_mask.sum()
                
                print(f"     GT{i+1}内anchor数: {inside_count}")
        
    except Exception as e:
        print(f"   ❌ 目标分配失败: {e}")
        import traceback
        traceback.print_exc()
        return None
    
    return {
        'num_pos': int(num_pos) if 'num_pos' in locals() else 0,
        'assigned_labels': assigned_labels,
        'assigned_scores': assigned_scores,
        'gt_bboxes_scaled': gt_bboxes_scaled,
        'anchor_points_scaled': anchor_points_scaled
    }

def compare_with_pytorch():
    """对比PyTorch版本的目标分配"""
    print(f"\n🔄 对比PyTorch版本的目标分配")
    print("=" * 80)
    
    # 这里可以加载PyTorch版本进行对比
    print(f"📋 PyTorch版本信息:")
    print(f"   模型参数: 5,617,930 (5.62M)")
    print(f"   输出格式: [1,5249,25]")
    print(f"   损失函数: VarifocalLoss + BboxLoss")
    print(f"   目标分配: ATSSAssigner + TaskAlignedAssigner")
    
    print(f"\n📋 Jittor版本信息:")
    print(f"   模型参数: 5,697,053 (5.70M)")
    print(f"   输出格式: [1,5249,25]")
    print(f"   损失函数: VarifocalLoss + BboxLoss")
    print(f"   目标分配: ATSSAssigner + TaskAlignedAssigner")
    
    print(f"\n🔍 关键差异分析:")
    print(f"   参数差异: +79,123 (1.4%)")
    print(f"   可能原因: 某些层的实现细节不同")
    print(f"   影响: 可能导致目标分配行为差异")

def main():
    print("🔍 深入对齐PyTorch版本")
    print("=" * 80)
    print("找到为什么参数一样但训练效果不同的根本原因")
    print("=" * 80)
    
    # 执行目标分配调试
    result = debug_target_assignment()
    
    # 对比PyTorch版本
    compare_with_pytorch()
    
    # 总结发现
    print(f"\n📊 调试总结:")
    print("=" * 80)
    
    if result and result['num_pos'] > 0:
        print(f"✅ 目标分配正常:")
        print(f"   正样本数: {result['num_pos']}")
        print(f"   分类损失应该>0")
        print(f"   问题可能在损失计算部分")
    else:
        print(f"❌ 目标分配有问题:")
        print(f"   正样本数: 0")
        print(f"   这就是分类损失为0的根本原因")
        print(f"   需要修复目标分配器")
    
    print(f"\n🔧 下一步行动:")
    if result and result['num_pos'] == 0:
        print(f"1. 深入调试ATSSAssigner的实现")
        print(f"2. 检查anchor生成是否正确")
        print(f"3. 对比PyTorch版本的目标分配逻辑")
        print(f"4. 修复目标分配器的bug")
    else:
        print(f"1. 深入调试损失计算过程")
        print(f"2. 检查VarifocalLoss的实现")
        print(f"3. 确保分类损失正确计算")
    
    # 更新问题追踪日志
    log_file = "PROBLEM_TRACKING_LOG.md"
    if os.path.exists(log_file):
        with open(log_file, 'a', encoding='utf-8') as f:
            f.write(f"\n## 🔍 深入调试记录 - {time.strftime('%Y-%m-%d %H:%M')}\n")
            f.write(f"### 目标分配调试结果\n")
            if result:
                f.write(f"- 正样本数: {result['num_pos']}\n")
                if result['num_pos'] == 0:
                    f.write(f"- **关键发现**: 没有正样本被分配，这是分类损失为0的根本原因\n")
                    f.write(f"- **下一步**: 修复目标分配器\n")
                else:
                    f.write(f"- **关键发现**: 目标分配正常，问题在损失计算\n")
                    f.write(f"- **下一步**: 调试损失计算过程\n")
            else:
                f.write(f"- **状态**: 调试失败，需要进一步分析\n")

if __name__ == "__main__":
    import time
    main()

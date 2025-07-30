#!/usr/bin/env python3
"""
深入调试损失函数问题
分析为什么过拟合失败
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
from yolov6.models.pytorch_aligned_losses import ComputeLoss

def debug_loss_function():
    """深入调试损失函数"""
    print(f"🔍 深入调试损失函数问题")
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
    print(f"   标签内容: {targets_tensor.data}")
    
    # 创建模型
    model = create_perfect_gold_yolo_model()
    model.train()
    
    # 前向传播
    print(f"\n🔄 模型前向传播:")
    outputs = model(img_tensor)
    
    print(f"   输出类型: {type(outputs)}")
    if isinstance(outputs, (list, tuple)):
        for i, output in enumerate(outputs):
            if hasattr(output, 'shape'):
                print(f"     输出{i}: {output.shape}")
                if i == 1:  # pred_scores
                    print(f"       分类分数范围: [{float(output.min()):.6f}, {float(output.max()):.6f}]")
                elif i == 2:  # pred_distri
                    print(f"       距离预测范围: [{float(output.min()):.6f}, {float(output.max()):.6f}]")
    
    # 创建损失函数
    print(f"\n💰 创建损失函数:")
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
    
    print(f"   损失权重: {loss_fn.loss_weight}")
    print(f"   FPN步长: {loss_fn.fpn_strides}")
    print(f"   使用DFL: {loss_fn.use_dfl}")
    
    # 详细分析损失计算过程
    print(f"\n🔍 详细分析损失计算:")
    
    try:
        # 手动调用损失函数的内部方法
        feats, pred_scores, pred_distri = outputs
        
        print(f"   特征列表长度: {len(feats)}")
        print(f"   预测分数形状: {pred_scores.shape}")
        print(f"   预测距离形状: {pred_distri.shape}")
        
        # 检查anchor points
        anchor_points, stride_tensor = loss_fn.generate_anchors(feats, loss_fn.fpn_strides, loss_fn.grid_cell_size, loss_fn.grid_cell_offset)
        print(f"   Anchor点数量: {anchor_points.shape[0]}")
        print(f"   Anchor点形状: {anchor_points.shape}")
        print(f"   步长张量形状: {stride_tensor.shape}")
        
        # 检查目标分配
        gt_labels = targets_tensor[..., 1]
        gt_bboxes = targets_tensor[..., 2:]
        
        print(f"   GT标签形状: {gt_labels.shape}")
        print(f"   GT框形状: {gt_bboxes.shape}")
        print(f"   GT标签内容: {gt_labels.data}")
        print(f"   GT框内容: {gt_bboxes.data}")
        
        # 转换GT框格式
        gt_bboxes_scaled = gt_bboxes * 500  # 缩放到图像尺寸
        print(f"   GT框缩放后: {gt_bboxes_scaled.data}")
        
        # 检查预测框解码
        pred_bboxes = loss_fn.bbox_decode(anchor_points, pred_distri)
        print(f"   预测框形状: {pred_bboxes.shape}")
        print(f"   预测框范围: [{float(pred_bboxes.min()):.2f}, {float(pred_bboxes.max()):.2f}]")
        
        # 检查目标分配结果
        try:
            assigned_labels, assigned_bboxes, assigned_scores = loss_fn.assigner(
                pred_scores.detach(), pred_bboxes.detach() * stride_tensor, 
                anchor_points * stride_tensor, gt_labels, gt_bboxes_scaled, mask_gt=None
            )
            
            print(f"   分配标签形状: {assigned_labels.shape}")
            print(f"   分配框形状: {assigned_bboxes.shape}")
            print(f"   分配分数形状: {assigned_scores.shape}")
            
            # 统计正样本数量
            pos_mask = assigned_labels > 0
            num_pos = pos_mask.sum()
            print(f"   正样本数量: {int(num_pos.data)}")
            
            if num_pos > 0:
                print(f"   正样本标签: {assigned_labels[pos_mask].data}")
                print(f"   正样本分数: {assigned_scores[pos_mask].data}")
            else:
                print(f"   ❌ 没有正样本！这是问题所在！")
                
        except Exception as e:
            print(f"   ❌ 目标分配失败: {e}")
            import traceback
            traceback.print_exc()
        
        # 计算完整损失
        print(f"\n💰 计算完整损失:")
        loss, loss_items = loss_fn(outputs, targets_tensor, epoch_num=0, step_num=0)
        
        print(f"   总损失: {float(loss.data):.6f}")
        print(f"   损失项: {[float(item.data) for item in loss_items]}")
        print(f"   损失项名称: ['IoU Loss', 'DFL Loss', 'Class Loss']")
        
        # 分析各项损失
        iou_loss, dfl_loss, cls_loss = loss_items
        print(f"\n📊 损失分析:")
        print(f"   IoU损失: {float(iou_loss.data):.6f} (权重: {loss_fn.loss_weight['iou']})")
        print(f"   DFL损失: {float(dfl_loss.data):.6f} (权重: {loss_fn.loss_weight['dfl']})")
        print(f"   分类损失: {float(cls_loss.data):.6f} (权重: {loss_fn.loss_weight['class']})")
        
        if float(cls_loss.data) == 0.0:
            print(f"   ⚠️ 分类损失为0，可能原因:")
            print(f"     1. 没有正样本被分配")
            print(f"     2. 分类损失计算有bug")
            print(f"     3. 标签格式不正确")
        
        if float(iou_loss.data) > 100:
            print(f"   ⚠️ IoU损失过高，可能原因:")
            print(f"     1. 预测框与GT框差距太大")
            print(f"     2. 坐标系不匹配")
            print(f"     3. anchor生成有问题")
        
    except Exception as e:
        print(f"❌ 损失计算失败: {e}")
        import traceback
        traceback.print_exc()

def main():
    print("🔍 损失函数深入调试")
    print("=" * 80)
    
    debug_loss_function()

if __name__ == "__main__":
    main()

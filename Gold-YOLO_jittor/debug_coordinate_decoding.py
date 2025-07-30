#!/usr/bin/env python3
"""
深入调试坐标解码的每一步
找出坐标异常的根本原因
"""

import os
import sys
import cv2
import numpy as np
import jittor as jt
from pathlib import Path
import time
import math

# 添加路径
sys.path.append('.')
sys.path.append('./yolov6')

from models.perfect_gold_yolo import create_perfect_gold_yolo_model
from yolov6.data.data_augment import letterbox
from yolov6.models.pytorch_aligned_losses import ComputeLoss
from yolov6.assigners.anchor_generator import generate_anchors
from yolov6.utils.general import dist2bbox

def debug_coordinate_decoding():
    """深入调试坐标解码的每一步"""
    print(f"🔍 深入调试坐标解码")
    print("=" * 80)
    
    # 准备数据
    img_path = "/home/kyc/project/GOLD-YOLO/2008_001420.jpg"
    original_img = cv2.imread(img_path)
    img_height, img_width = original_img.shape[:2]
    
    # 预处理图像
    img_size = 640
    img = letterbox(original_img, new_shape=img_size, stride=32, auto=False)[0]
    img_tensor_input = img.transpose((2, 0, 1))[::-1]
    img_tensor_input = np.ascontiguousarray(img_tensor_input)
    img_tensor_input = img_tensor_input.astype(np.float32) / 255.0
    img_tensor = jt.array(img_tensor_input).unsqueeze(0)
    
    print(f"📊 数据准备:")
    print(f"   原始图像尺寸: {img_width}x{img_height}")
    print(f"   预处理后尺寸: {img.shape}")
    
    # 创建模型
    model = create_perfect_gold_yolo_model()
    
    # 快速训练几轮
    model.train()
    targets = [[0, 11, 0.814, 0.400, 0.111, 0.208]]  # 一个dog目标
    targets_tensor = jt.array(targets, dtype=jt.float32).unsqueeze(0)
    
    loss_fn = ComputeLoss(
        num_classes=20,
        ori_img_size=img_size,
        warmup_epoch=0,
        use_dfl=False,
        reg_max=0,
        fpn_strides=[8, 16, 32],
        grid_cell_size=5.0,
        grid_cell_offset=0.5,
        loss_weight={'class': 1.0, 'iou': 2.5, 'dfl': 0.5}
    )
    
    optimizer = jt.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
    
    print(f"\n🔧 快速训练5轮:")
    for epoch in range(5):
        outputs = model(img_tensor)
        loss, loss_items = loss_fn(outputs, targets_tensor, epoch_num=epoch, step_num=epoch)
        optimizer.zero_grad()
        optimizer.backward(loss)
        optimizer.step()
        print(f"   轮次 {epoch}: 损失={float(loss):.6f}")
    
    # 切换到推理模式并手动执行每一步
    print(f"\n🔍 手动执行推理的每一步:")
    model.eval()
    
    with jt.no_grad():
        # 1. 获取特征
        print(f"\n1️⃣ 获取backbone和neck特征:")
        x = img_tensor
        
        # 通过backbone
        backbone_features = []
        for i, layer in enumerate(model.backbone.backbone):
            x = layer(x)
            if i in [2, 4, 6]:  # 假设这些是输出层
                backbone_features.append(x)
                print(f"   Backbone层{i}: {x.shape}")
        
        # 通过neck
        neck_features = model.neck(backbone_features)
        print(f"   Neck输出: {[f.shape for f in neck_features]}")
        
        # 2. 生成anchor_points
        print(f"\n2️⃣ 生成anchor_points:")
        anchor_points, stride_tensor = generate_anchors(
            neck_features, [8, 16, 32], 
            grid_cell_size=5.0, grid_cell_offset=0.5, 
            device=None, is_eval=True, mode='af'
        )
        
        print(f"   anchor_points形状: {anchor_points.shape}")
        print(f"   stride_tensor形状: {stride_tensor.shape}")
        print(f"   anchor_points范围: [{float(anchor_points.min()):.2f}, {float(anchor_points.max()):.2f}]")
        print(f"   stride_tensor范围: [{float(stride_tensor.min()):.2f}, {float(stride_tensor.max()):.2f}]")
        
        # 检查前几个anchor_points
        print(f"   前5个anchor_points:")
        for i in range(5):
            x, y = anchor_points[i]
            stride = stride_tensor[i, 0]
            print(f"     anchor{i}: ({float(x):.2f}, {float(y):.2f}), stride={float(stride):.0f}")
        
        # 3. 通过head获取原始输出
        print(f"\n3️⃣ 通过head获取原始输出:")
        cls_score_list = []
        reg_dist_list = []
        
        for i in range(len(neck_features)):
            x_i = neck_features[i]
            b, _, h, w = x_i.shape
            l = h * w
            
            # 通过head的各个层
            x_i = model.head.stems[i](x_i)
            cls_feat = model.head.cls_convs[i](x_i)
            cls_output = model.head.cls_preds[i](cls_feat)
            reg_feat = model.head.reg_convs[i](x_i)
            reg_output = model.head.reg_preds[i](reg_feat)
            
            # 应用sigmoid
            cls_output = jt.sigmoid(cls_output)
            cls_score_list.append(cls_output.reshape([b, model.head.nc, l]))
            reg_dist_list.append(reg_output.reshape([b, 4, l]))
            
            print(f"   Head层{i}: 特征{x_i.shape} -> 分类{cls_output.shape}, 回归{reg_output.shape}")
        
        cls_score_list = jt.concat(cls_score_list, dim=-1).permute(0, 2, 1)
        reg_dist_list = jt.concat(reg_dist_list, dim=-1).permute(0, 2, 1)
        
        print(f"   合并后分类: {cls_score_list.shape}")
        print(f"   合并后回归: {reg_dist_list.shape}")
        print(f"   回归输出范围: [{float(reg_dist_list.min()):.2f}, {float(reg_dist_list.max()):.2f}]")
        
        # 4. 应用dist2bbox
        print(f"\n4️⃣ 应用dist2bbox转换:")
        print(f"   输入reg_dist_list: {reg_dist_list.shape}, 范围=[{float(reg_dist_list.min()):.2f}, {float(reg_dist_list.max()):.2f}]")
        print(f"   输入anchor_points: {anchor_points.shape}, 范围=[{float(anchor_points.min()):.2f}, {float(anchor_points.max()):.2f}]")
        
        pred_bboxes = dist2bbox(reg_dist_list, anchor_points, box_format='xywh')
        print(f"   dist2bbox输出: {pred_bboxes.shape}, 范围=[{float(pred_bboxes.min()):.2f}, {float(pred_bboxes.max()):.2f}]")
        
        # 检查前几个bbox
        print(f"   前5个bbox (dist2bbox后):")
        for i in range(5):
            x, y, w, h = pred_bboxes[0, i]
            print(f"     bbox{i}: ({float(x):.2f}, {float(y):.2f}, {float(w):.2f}, {float(h):.2f})")
        
        # 5. 乘以stride_tensor
        print(f"\n5️⃣ 乘以stride_tensor:")
        print(f"   乘法前: 范围=[{float(pred_bboxes.min()):.2f}, {float(pred_bboxes.max()):.2f}]")
        
        pred_bboxes_scaled = pred_bboxes * stride_tensor
        print(f"   乘法后: 范围=[{float(pred_bboxes_scaled.min()):.2f}, {float(pred_bboxes_scaled.max()):.2f}]")
        
        # 检查前几个缩放后的bbox
        print(f"   前5个bbox (缩放后):")
        for i in range(5):
            x, y, w, h = pred_bboxes_scaled[0, i]
            stride = stride_tensor[i, 0]
            print(f"     bbox{i}: ({float(x):.2f}, {float(y):.2f}, {float(w):.2f}, {float(h):.2f}), stride={float(stride):.0f}")
        
        # 6. 检查是否超出合理范围
        print(f"\n6️⃣ 检查坐标合理性:")
        max_coord = float(pred_bboxes_scaled.max())
        min_coord = float(pred_bboxes_scaled.min())
        
        print(f"   最大坐标: {max_coord:.2f}")
        print(f"   最小坐标: {min_coord:.2f}")
        print(f"   图像尺寸: {img_size}x{img_size}")
        
        if max_coord > img_size * 10:
            print(f"   ❌ 坐标过大！可能stride缩放有问题")
        elif min_coord < -img_size:
            print(f"   ❌ 坐标过小！可能有偏移问题")
        else:
            print(f"   ✅ 坐标在合理范围内")
        
        # 7. 分析stride_tensor的问题
        print(f"\n7️⃣ 分析stride_tensor:")
        unique_strides = jt.unique(stride_tensor)
        print(f"   唯一的stride值: {[float(s) for s in unique_strides]}")
        
        # 检查stride_tensor的形状是否正确
        print(f"   stride_tensor形状: {stride_tensor.shape}")
        print(f"   pred_bboxes形状: {pred_bboxes.shape}")
        
        # 检查广播是否正确
        if stride_tensor.shape[0] != pred_bboxes.shape[1]:
            print(f"   ❌ 形状不匹配！stride_tensor[0]={stride_tensor.shape[0]}, pred_bboxes[1]={pred_bboxes.shape[1]}")
        else:
            print(f"   ✅ 形状匹配")
        
        return pred_bboxes_scaled

def main():
    print("🔍 深入调试坐标解码")
    print("=" * 80)
    
    try:
        pred_bboxes = debug_coordinate_decoding()
        
        print(f"\n📊 调试总结:")
        print(f"   最终坐标范围: [{float(pred_bboxes.min()):.2f}, {float(pred_bboxes.max()):.2f}]")
        
    except Exception as e:
        print(f"\n❌ 调试异常: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()

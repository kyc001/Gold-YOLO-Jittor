#!/usr/bin/env python3
"""
深入调试数值
对比每一步的具体数值
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

def deep_debug_values():
    """深入调试数值"""
    print(f"🔍 深入调试数值")
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
    print(f"   图像尺寸: {img_size}x{img_size}")
    
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
    
    print(f"\n🔧 快速训练3轮:")
    for epoch in range(3):
        outputs = model(img_tensor)
        loss, loss_items = loss_fn(outputs, targets_tensor, epoch_num=epoch, step_num=epoch)
        optimizer.zero_grad()
        optimizer.backward(loss)
        optimizer.step()
        print(f"   轮次 {epoch}: 损失={float(loss):.6f}")
    
    # 切换到推理模式并手动执行每一步
    print(f"\n🔍 手动执行推理并检查每一步数值:")
    model.eval()
    
    with jt.no_grad():
        # 1. 获取neck特征
        print(f"\n1️⃣ 获取neck特征:")
        neck_features = model.neck(model.backbone(img_tensor))
        
        for i, feat in enumerate(neck_features):
            print(f"   特征{i}: {feat.shape}, 范围=[{float(feat.min()):.6f}, {float(feat.max()):.6f}]")
        
        # 2. 生成anchor_points
        print(f"\n2️⃣ 生成anchor_points:")
        anchor_points, stride_tensor = generate_anchors(
            neck_features, [8, 16, 32], 
            grid_cell_size=5.0, grid_cell_offset=0.5, 
            device=None, is_eval=True, mode='af'
        )
        
        print(f"   anchor_points: {anchor_points.shape}, 范围=[{float(anchor_points.min()):.2f}, {float(anchor_points.max()):.2f}]")
        print(f"   stride_tensor: {stride_tensor.shape}, 范围=[{float(stride_tensor.min()):.0f}, {float(stride_tensor.max()):.0f}]")
        
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
            
            print(f"   处理特征{i}: {x_i.shape} -> 网格{h}x{w} = {l}个点")
            
            # 通过head的各个层
            x_i = model.head.stems[i](x_i)
            cls_feat = model.head.cls_convs[i](x_i)
            cls_output = model.head.cls_preds[i](cls_feat)
            reg_feat = model.head.reg_convs[i](x_i)
            reg_output = model.head.reg_preds[i](reg_feat)
            
            print(f"     原始reg_output: {reg_output.shape}, 范围=[{float(reg_output.min()):.6f}, {float(reg_output.max()):.6f}]")
            
            # 应用sigmoid
            cls_output = jt.sigmoid(cls_output)
            cls_score_list.append(cls_output.reshape([b, model.head.nc, l]))
            reg_dist_list.append(reg_output.reshape([b, 4, l]))
            
            print(f"     重塑后reg_output: {reg_output.reshape([b, 4, l]).shape}")
        
        cls_score_list = jt.concat(cls_score_list, dim=-1).permute(0, 2, 1)
        reg_dist_list = jt.concat(reg_dist_list, dim=-1).permute(0, 2, 1)
        
        print(f"   合并后reg_dist_list: {reg_dist_list.shape}, 范围=[{float(reg_dist_list.min()):.6f}, {float(reg_dist_list.max()):.6f}]")
        
        # 检查前几个reg_dist值
        print(f"   前5个reg_dist值:")
        for i in range(5):
            l, t, r, b = reg_dist_list[0, i]
            print(f"     reg_dist{i}: l={float(l):.6f}, t={float(t):.6f}, r={float(r):.6f}, b={float(b):.6f}")
        
        # 4. 应用dist2bbox
        print(f"\n4️⃣ 应用dist2bbox转换:")
        
        # 手动执行dist2bbox的每一步
        print(f"   输入检查:")
        print(f"     reg_dist_list: {reg_dist_list.shape}, 范围=[{float(reg_dist_list.min()):.6f}, {float(reg_dist_list.max()):.6f}]")
        print(f"     anchor_points: {anchor_points.shape}, 范围=[{float(anchor_points.min()):.2f}, {float(anchor_points.max()):.2f}]")
        
        # 分割距离
        lt, rb = jt.split(reg_dist_list, 2, -1)
        print(f"   分割后:")
        print(f"     lt: {lt.shape}, 范围=[{float(lt.min()):.6f}, {float(lt.max()):.6f}]")
        print(f"     rb: {rb.shape}, 范围=[{float(rb.min()):.6f}, {float(rb.max()):.6f}]")
        
        # 计算x1y1和x2y2
        x1y1 = anchor_points - lt
        x2y2 = anchor_points + rb
        print(f"   计算坐标:")
        print(f"     x1y1: {x1y1.shape}, 范围=[{float(x1y1.min()):.6f}, {float(x1y1.max()):.6f}]")
        print(f"     x2y2: {x2y2.shape}, 范围=[{float(x2y2.min()):.6f}, {float(x2y2.max()):.6f}]")
        
        # 转换为xywh格式
        c_xy = (x1y1 + x2y2) / 2
        wh = x2y2 - x1y1
        pred_bboxes = jt.concat([c_xy, wh], -1)
        
        print(f"   转换为xywh:")
        print(f"     c_xy: {c_xy.shape}, 范围=[{float(c_xy.min()):.6f}, {float(c_xy.max()):.6f}]")
        print(f"     wh: {wh.shape}, 范围=[{float(wh.min()):.6f}, {float(wh.max()):.6f}]")
        print(f"     pred_bboxes: {pred_bboxes.shape}, 范围=[{float(pred_bboxes.min()):.6f}, {float(pred_bboxes.max()):.6f}]")
        
        # 检查前几个bbox
        print(f"   前5个bbox (dist2bbox后):")
        for i in range(5):
            x, y, w, h = pred_bboxes[0, i]
            anchor_x, anchor_y = anchor_points[i]
            l_val, t_val = lt[0, i]
            r_val, b_val = rb[0, i]
            print(f"     bbox{i}: anchor=({float(anchor_x):.2f},{float(anchor_y):.2f}), lt=({float(l_val):.6f},{float(t_val):.6f}), rb=({float(r_val):.6f},{float(b_val):.6f}) -> ({float(x):.6f},{float(y):.6f},{float(w):.6f},{float(h):.6f})")
        
        # 5. 乘以stride_tensor
        print(f"\n5️⃣ 乘以stride_tensor:")
        print(f"   乘法前pred_bboxes: 范围=[{float(pred_bboxes.min()):.6f}, {float(pred_bboxes.max()):.6f}]")
        print(f"   stride_tensor: 范围=[{float(stride_tensor.min()):.0f}, {float(stride_tensor.max()):.0f}]")
        
        pred_bboxes_scaled = pred_bboxes * stride_tensor
        print(f"   乘法后pred_bboxes: 范围=[{float(pred_bboxes_scaled.min()):.6f}, {float(pred_bboxes_scaled.max()):.6f}]")
        
        # 检查前几个缩放后的bbox
        print(f"   前5个bbox (缩放后):")
        for i in range(5):
            x, y, w, h = pred_bboxes_scaled[0, i]
            stride = stride_tensor[i, 0]
            x_before, y_before, w_before, h_before = pred_bboxes[0, i]
            print(f"     bbox{i}: 缩放前=({float(x_before):.6f},{float(y_before):.6f},{float(w_before):.6f},{float(h_before):.6f}) * {float(stride):.0f} = ({float(x):.6f},{float(y):.6f},{float(w):.6f},{float(h):.6f})")
        
        # 6. 分析问题
        print(f"\n6️⃣ 问题分析:")
        
        # 检查reg_dist_list的值是否异常
        max_reg = float(reg_dist_list.max())
        min_reg = float(reg_dist_list.min())
        
        print(f"   reg_dist_list统计:")
        print(f"     最大值: {max_reg:.6f}")
        print(f"     最小值: {min_reg:.6f}")
        print(f"     绝对值最大: {max(abs(max_reg), abs(min_reg)):.6f}")
        
        if max(abs(max_reg), abs(min_reg)) > 1000:
            print(f"   ❌ reg_dist_list值异常大！这是问题根源！")
            print(f"   原因可能是:")
            print(f"     1. 模型权重初始化问题")
            print(f"     2. 训练不稳定导致权重爆炸")
            print(f"     3. 梯度爆炸")
        else:
            print(f"   ✅ reg_dist_list值正常")
        
        return pred_bboxes_scaled

def main():
    print("🔍 深入调试数值")
    print("=" * 80)
    
    try:
        pred_bboxes = deep_debug_values()
        
        print(f"\n📊 调试总结:")
        print(f"   最终坐标范围: [{float(pred_bboxes.min()):.6f}, {float(pred_bboxes.max()):.6f}]")
        
    except Exception as e:
        print(f"\n❌ 调试异常: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()

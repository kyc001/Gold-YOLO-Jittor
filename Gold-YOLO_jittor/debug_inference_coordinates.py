#!/usr/bin/env python3
"""
调试推理坐标问题
分析为什么预测框坐标异常
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

def debug_inference_coordinates():
    """调试推理坐标问题"""
    print(f"🔍 调试推理坐标问题")
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
    print(f"   输入张量: {img_tensor.shape}")
    
    # 创建模型
    model = create_perfect_gold_yolo_model()
    
    # 先训练几轮让模型有合理的输出
    print(f"\n🔧 快速训练几轮:")
    model.train()
    
    # 准备标签
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
    
    for epoch in range(10):
        outputs = model(img_tensor)
        loss, loss_items = loss_fn(outputs, targets_tensor, epoch_num=epoch, step_num=epoch)
        optimizer.zero_grad()
        optimizer.backward(loss)
        optimizer.step()
        
        if epoch % 5 == 0:
            print(f"   轮次 {epoch}: 损失={float(loss):.6f}")
    
    # 切换到推理模式
    print(f"\n🔍 切换到推理模式:")
    model.eval()
    
    with jt.no_grad():
        # 推理
        outputs = model(img_tensor)
        
        print(f"   推理输出形状: {outputs.shape}")
        print(f"   输出范围: [{float(outputs.min()):.6f}, {float(outputs.max()):.6f}]")
        
        # 分析输出的各个部分
        batch_size, num_anchors, num_features = outputs.shape
        print(f"   批次大小: {batch_size}")
        print(f"   anchor数量: {num_anchors}")
        print(f"   特征数量: {num_features}")
        
        # 分解输出
        pred_bboxes = outputs[..., :4]      # [1, 8400, 4] 坐标
        pred_obj = outputs[..., 4:5]        # [1, 8400, 1] objectness
        pred_cls = outputs[..., 5:]         # [1, 8400, 20] 类别
        
        print(f"\n📊 输出分析:")
        print(f"   预测框: {pred_bboxes.shape}, 范围=[{float(pred_bboxes.min()):.2f}, {float(pred_bboxes.max()):.2f}]")
        print(f"   objectness: {pred_obj.shape}, 范围=[{float(pred_obj.min()):.6f}, {float(pred_obj.max()):.6f}]")
        print(f"   类别分数: {pred_cls.shape}, 范围=[{float(pred_cls.min()):.6f}, {float(pred_cls.max()):.6f}]")
        
        # 检查坐标格式
        print(f"\n🔍 坐标格式分析:")
        sample_boxes = pred_bboxes[0, :10]  # 前10个框
        for i, box in enumerate(sample_boxes):
            x, y, w, h = box
            print(f"   框{i+1}: x={float(x):.2f}, y={float(y):.2f}, w={float(w):.2f}, h={float(h):.2f}")
        
        # 检查是否是xywh格式还是xyxy格式
        print(f"\n🔍 坐标格式判断:")
        if float(pred_bboxes.max()) > img_size * 2:
            print(f"   ❌ 坐标值过大，可能有缩放问题")
        elif float(pred_bboxes.min()) < -img_size:
            print(f"   ❌ 坐标值过小，可能有偏移问题")
        else:
            print(f"   ✅ 坐标值在合理范围内")
        
        # 检查objectness和类别分数
        print(f"\n🔍 置信度分析:")
        max_obj = float(pred_obj.max())
        max_cls = float(pred_cls.max())
        print(f"   最大objectness: {max_obj:.6f}")
        print(f"   最大类别分数: {max_cls:.6f}")
        
        # 计算总置信度
        total_conf = pred_obj * pred_cls.max(dim=-1, keepdim=True)[0]
        max_total_conf = float(total_conf.max())
        print(f"   最大总置信度: {max_total_conf:.6f}")
        
        # 找到高置信度的预测
        high_conf_mask = total_conf.squeeze(-1) > 0.1
        high_conf_indices = jt.where(high_conf_mask)
        
        if len(high_conf_indices[1]) > 0:
            print(f"\n🎯 高置信度预测 (>0.1):")
            for i in range(min(5, len(high_conf_indices[1]))):
                idx = high_conf_indices[1][i]
                box = pred_bboxes[0, idx]
                obj = pred_obj[0, idx, 0]
                cls_scores = pred_cls[0, idx]
                max_cls_idx = jt.argmax(cls_scores)[0]
                max_cls_score = cls_scores[max_cls_idx]
                
                x, y, w, h = box
                print(f"   预测{i+1}: 坐标=({float(x):.1f},{float(y):.1f},{float(w):.1f},{float(h):.1f}), obj={float(obj):.3f}, cls={int(max_cls_idx)}({float(max_cls_score):.3f})")
        else:
            print(f"\n❌ 没有高置信度预测")
        
        # 检查坐标是否需要转换
        print(f"\n🔧 坐标转换测试:")
        
        # 假设是xywh格式，转换为xyxy
        x_center, y_center, width, height = pred_bboxes[0, 0]
        x1 = float(x_center - width / 2)
        y1 = float(y_center - height / 2)
        x2 = float(x_center + width / 2)
        y2 = float(y_center + height / 2)
        
        print(f"   第一个框 xywh->xyxy: ({x1:.1f},{y1:.1f}) -> ({x2:.1f},{y2:.1f})")
        
        # 检查是否需要缩放到原始图像
        scale_x = img_width / img_size
        scale_y = img_height / img_size
        
        x1_scaled = x1 * scale_x
        y1_scaled = y1 * scale_y
        x2_scaled = x2 * scale_x
        y2_scaled = y2 * scale_y
        
        print(f"   缩放到原始图像: ({x1_scaled:.1f},{y1_scaled:.1f}) -> ({x2_scaled:.1f},{y2_scaled:.1f})")
        print(f"   缩放因子: x={scale_x:.3f}, y={scale_y:.3f}")
        
        if 0 <= x1_scaled <= img_width and 0 <= y1_scaled <= img_height and 0 <= x2_scaled <= img_width and 0 <= y2_scaled <= img_height:
            print(f"   ✅ 缩放后坐标在图像范围内")
        else:
            print(f"   ❌ 缩放后坐标仍然超出图像范围")
        
        return outputs

def main():
    print("🔍 调试推理坐标问题")
    print("=" * 80)
    
    try:
        outputs = debug_inference_coordinates()
        
        print(f"\n📊 调试总结:")
        print(f"   输出形状: {outputs.shape}")
        print(f"   输出范围: [{float(outputs.min()):.2f}, {float(outputs.max()):.2f}]")
        
    except Exception as e:
        print(f"\n❌ 调试异常: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
简化的坐标调试
直接修复effidehead中的坐标解码问题
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

def simple_coordinate_debug():
    """简化的坐标调试"""
    print(f"🔧 简化的坐标调试")
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
    
    # 切换到推理模式
    print(f"\n🔍 推理模式调试:")
    model.eval()
    
    with jt.no_grad():
        # 直接调用模型推理
        outputs = model(img_tensor)
        
        print(f"   推理输出形状: {outputs.shape}")
        print(f"   输出范围: [{float(outputs.min()):.2f}, {float(outputs.max()):.2f}]")
        
        # 分解输出
        pred_bboxes = outputs[..., :4]      # [1, 8400, 4] 坐标
        pred_obj = outputs[..., 4:5]        # [1, 8400, 1] objectness
        pred_cls = outputs[..., 5:]         # [1, 8400, 20] 类别
        
        print(f"   预测框: {pred_bboxes.shape}, 范围=[{float(pred_bboxes.min()):.2f}, {float(pred_bboxes.max()):.2f}]")
        print(f"   objectness: {pred_obj.shape}, 范围=[{float(pred_obj.min()):.6f}, {float(pred_obj.max()):.6f}]")
        print(f"   类别分数: {pred_cls.shape}, 范围=[{float(pred_cls.min()):.6f}, {float(pred_cls.max()):.6f}]")
        
        # 检查前几个预测框
        print(f"\n🔍 前10个预测框分析:")
        for i in range(10):
            x, y, w, h = pred_bboxes[0, i]
            obj = pred_obj[0, i, 0]
            max_cls_score = float(pred_cls[0, i].max())
            
            print(f"   框{i+1}: ({float(x):.1f}, {float(y):.1f}, {float(w):.1f}, {float(h):.1f}), obj={float(obj):.3f}, max_cls={max_cls_score:.3f}")
        
        # 检查坐标是否合理
        print(f"\n🔍 坐标合理性检查:")
        max_coord = float(pred_bboxes.max())
        min_coord = float(pred_bboxes.min())
        
        print(f"   最大坐标: {max_coord:.2f}")
        print(f"   最小坐标: {min_coord:.2f}")
        print(f"   图像尺寸: {img_size}x{img_size}")
        
        if abs(max_coord) > img_size * 100:
            print(f"   ❌ 坐标异常！需要修复")
            
            # 尝试修复：将坐标限制在合理范围内
            print(f"\n🔧 尝试修复坐标:")
            
            # 方法1：简单裁剪
            pred_bboxes_clipped = jt.clamp(pred_bboxes, -img_size, img_size * 2)
            print(f"   裁剪后范围: [{float(pred_bboxes_clipped.min()):.2f}, {float(pred_bboxes_clipped.max()):.2f}]")
            
            # 方法2：缩放到合理范围
            scale_factor = img_size / max(abs(max_coord), abs(min_coord))
            pred_bboxes_scaled = pred_bboxes * scale_factor
            print(f"   缩放因子: {scale_factor:.6f}")
            print(f"   缩放后范围: [{float(pred_bboxes_scaled.min()):.2f}, {float(pred_bboxes_scaled.max()):.2f}]")
            
            # 使用缩放后的坐标
            fixed_outputs = jt.concat([
                pred_bboxes_scaled,
                pred_obj,
                pred_cls
            ], dim=-1)
            
            print(f"   修复后输出: {fixed_outputs.shape}")
            
            return fixed_outputs
        else:
            print(f"   ✅ 坐标在合理范围内")
            return outputs

def draw_bbox(img, bbox, label, conf, color=(0, 255, 0)):
    """绘制边界框"""
    x1, y1, x2, y2 = bbox
    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
    
    # 确保坐标在图像范围内
    img_h, img_w = img.shape[:2]
    x1 = max(0, min(x1, img_w-1))
    y1 = max(0, min(y1, img_h-1))
    x2 = max(0, min(x2, img_w-1))
    y2 = max(0, min(y2, img_h-1))
    
    if x2 > x1 and y2 > y1:  # 只有当框有效时才绘制
        # 绘制边界框
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
        
        # 绘制标签
        VOC_CLASSES = [
            'aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car', 'cat',
            'chair', 'cow', 'diningtable', 'dog', 'horse', 'motorbike', 'person',
            'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor'
        ]
        
        if 0 <= label < len(VOC_CLASSES):
            label_text = f"{VOC_CLASSES[int(label)]}: {conf:.2f}"
            label_size = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
            cv2.rectangle(img, (x1, y1 - label_size[1] - 10), (x1 + label_size[0], y1), color, -1)
            cv2.putText(img, label_text, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

def visualize_fixed_results(outputs):
    """可视化修复后的结果"""
    print(f"\n🎨 可视化修复后的结果:")
    
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
    
    # 读取图像
    original_img = cv2.imread(img_path)
    img_height, img_width = original_img.shape[:2]
    vis_img = original_img.copy()
    
    # 绘制GT框 (绿色)
    print(f"   绘制GT框:")
    for i, ann in enumerate(annotations):
        cls_id, x_center, y_center, width, height = ann
        
        # 转换为像素坐标
        x_center_px = x_center * img_width
        y_center_px = y_center * img_height
        width_px = width * img_width
        height_px = height * img_height
        
        x1 = x_center_px - width_px / 2
        y1 = y_center_px - height_px / 2
        x2 = x_center_px + width_px / 2
        y2 = y_center_px + height_px / 2
        
        draw_bbox(vis_img, [x1, y1, x2, y2], cls_id, 1.0, color=(0, 255, 0))
        print(f"     GT{i+1}: {x1:.0f},{y1:.0f} -> {x2:.0f},{y2:.0f}")
    
    # 简单的预测框处理
    pred_bboxes = outputs[..., :4]      # [1, 8400, 4] 坐标
    pred_obj = outputs[..., 4:5]        # [1, 8400, 1] objectness
    pred_cls = outputs[..., 5:]         # [1, 8400, 20] 类别
    
    # 找到高置信度的预测
    total_conf = pred_obj * pred_cls.max(dim=-1, keepdim=True)[0]
    high_conf_indices = jt.where(total_conf.squeeze(-1) > 0.1)
    
    print(f"   高置信度预测数量: {len(high_conf_indices[1]) if len(high_conf_indices) > 1 else 0}")
    
    # 绘制前几个预测框 (红色)
    if len(high_conf_indices) > 1 and len(high_conf_indices[1]) > 0:
        print(f"   绘制预测框:")
        scale_x = img_width / 640
        scale_y = img_height / 640
        
        for i in range(min(5, len(high_conf_indices[1]))):
            idx = int(high_conf_indices[1][i])
            
            # 获取坐标 (假设是xywh格式)
            x_center, y_center, width, height = pred_bboxes[0, idx]
            
            # 转换为xyxy格式并缩放
            x1 = (float(x_center) - float(width)/2) * scale_x
            y1 = (float(y_center) - float(height)/2) * scale_y
            x2 = (float(x_center) + float(width)/2) * scale_x
            y2 = (float(y_center) + float(height)/2) * scale_y
            
            # 获取类别和置信度
            obj_conf = float(pred_obj[0, idx, 0])
            cls_scores = pred_cls[0, idx]
            max_cls_idx = int(jt.argmax(cls_scores, dim=0)[0])
            max_cls_score = float(cls_scores[max_cls_idx])
            total_conf_val = obj_conf * max_cls_score
            
            draw_bbox(vis_img, [x1, y1, x2, y2], max_cls_idx, total_conf_val, color=(0, 0, 255))
            print(f"     预测{i+1}: {x1:.0f},{y1:.0f} -> {x2:.0f},{y2:.0f}, conf={total_conf_val:.3f}")
    
    # 保存结果
    output_path = "coordinate_debug_result.jpg"
    cv2.imwrite(output_path, vis_img)
    print(f"   结果已保存: {output_path}")

def main():
    print("🔧 简化的坐标调试")
    print("=" * 80)
    
    try:
        outputs = simple_coordinate_debug()
        
        print(f"\n📊 调试总结:")
        print(f"   输出形状: {outputs.shape}")
        print(f"   输出范围: [{float(outputs.min()):.2f}, {float(outputs.max()):.2f}]")
        
        # 可视化结果
        visualize_fixed_results(outputs)
        
        print(f"\n✅ 坐标调试完成！")
        
    except Exception as e:
        print(f"\n❌ 调试异常: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()

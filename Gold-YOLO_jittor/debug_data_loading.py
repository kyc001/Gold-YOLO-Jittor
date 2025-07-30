#!/usr/bin/env python3
"""
深入调试数据加载问题
检查图像和标注数据是否正确读取
"""

import os
import sys
import cv2
import numpy as np
import jittor as jt
from pathlib import Path
import matplotlib.pyplot as plt

# 添加路径
sys.path.append('.')
sys.path.append('./yolov6')

from yolov6.data.data_augment import letterbox

# VOC数据集类别名称
VOC_CLASSES = [
    'aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car', 'cat', 'chair', 'cow',
    'diningtable', 'dog', 'horse', 'motorbike', 'person', 'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor'
]

def debug_data_loading():
    """深入调试数据加载问题"""
    print(f"🔍 深入调试数据加载问题")
    print("=" * 80)
    
    # 数据路径
    label_file = "/home/kyc/project/GOLD-YOLO/2008_001420.txt"
    img_path = "/home/kyc/project/GOLD-YOLO/2008_001420.jpg"
    
    print(f"📁 数据路径:")
    print(f"   图像: {img_path}")
    print(f"   标注: {label_file}")
    print(f"   图像存在: {os.path.exists(img_path)}")
    print(f"   标注存在: {os.path.exists(label_file)}")
    
    # 1. 读取并分析原始标注
    print(f"\n📋 原始标注分析:")
    annotations = []
    with open(label_file, 'r') as f:
        for i, line in enumerate(f):
            line = line.strip()
            if line:
                parts = line.split()
                print(f"   第{i+1}行: {parts}")
                if len(parts) >= 5:
                    cls_id = int(parts[0])
                    x_center = float(parts[1])
                    y_center = float(parts[2])
                    width = float(parts[3])
                    height = float(parts[4])
                    annotations.append([cls_id, x_center, y_center, width, height])
                    print(f"     解析: 类别={VOC_CLASSES[cls_id]}, 中心=({x_center:.3f},{y_center:.3f}), 尺寸=({width:.3f},{height:.3f})")
    
    print(f"   总标注数: {len(annotations)}")
    
    # 2. 读取并分析原始图像
    print(f"\n🖼️ 原始图像分析:")
    original_img = cv2.imread(img_path)
    if original_img is None:
        print(f"❌ 无法读取图像: {img_path}")
        return
    
    img_height, img_width = original_img.shape[:2]
    print(f"   图像尺寸: {img_width} x {img_height}")
    print(f"   图像通道: {original_img.shape[2]}")
    print(f"   图像类型: {original_img.dtype}")
    print(f"   像素范围: [{original_img.min()}, {original_img.max()}]")
    
    # 3. 转换标注为像素坐标并验证
    print(f"\n📐 标注坐标转换:")
    gt_boxes = []
    gt_classes = []
    
    for i, ann in enumerate(annotations):
        cls_id, x_center, y_center, width, height = ann
        
        # 转换为像素坐标
        x1 = int((x_center - width/2) * img_width)
        y1 = int((y_center - height/2) * img_height)
        x2 = int((x_center + width/2) * img_width)
        y2 = int((y_center + height/2) * img_height)
        
        gt_boxes.append([x1, y1, x2, y2])
        gt_classes.append(cls_id)
        
        print(f"   标注{i+1}: {VOC_CLASSES[cls_id]}")
        print(f"     归一化: 中心({x_center:.3f},{y_center:.3f}), 尺寸({width:.3f},{height:.3f})")
        print(f"     像素坐标: ({x1},{y1}) -> ({x2},{y2})")
        print(f"     框尺寸: {x2-x1} x {y2-y1}")
        
        # 验证坐标是否合理
        if x1 < 0 or y1 < 0 or x2 >= img_width or y2 >= img_height:
            print(f"     ⚠️ 坐标超出图像边界！")
        if x2 <= x1 or y2 <= y1:
            print(f"     ❌ 无效的框尺寸！")
    
    # 4. 图像预处理分析
    print(f"\n🔄 图像预处理分析:")
    img_resized = letterbox(original_img, new_shape=640, stride=32, auto=False)[0]
    print(f"   预处理后尺寸: {img_resized.shape}")
    
    # 转换为模型输入格式
    img_tensor_input = img_resized.transpose((2, 0, 1))[::-1]  # HWC -> CHW, BGR -> RGB
    img_tensor_input = np.ascontiguousarray(img_tensor_input)
    img_tensor_input = img_tensor_input.astype(np.float32) / 255.0
    
    print(f"   张量形状: {img_tensor_input.shape}")
    print(f"   张量类型: {img_tensor_input.dtype}")
    print(f"   像素范围: [{img_tensor_input.min():.3f}, {img_tensor_input.max():.3f}]")
    
    img_tensor = jt.array(img_tensor_input).unsqueeze(0)
    print(f"   Jittor张量形状: {img_tensor.shape}")
    
    # 5. 标签张量分析
    print(f"\n🏷️ 标签张量分析:")
    targets = []
    for ann in annotations:
        cls_id, x_center, y_center, width, height = ann
        targets.append([0, cls_id, x_center, y_center, width, height])  # [batch_idx, cls, x, y, w, h]
    
    targets_tensor = jt.array(targets, dtype=jt.float32).unsqueeze(0)
    print(f"   标签张量形状: {targets_tensor.shape}")
    print(f"   标签张量内容:")
    for i, target in enumerate(targets):
        print(f"     目标{i+1}: batch={target[0]}, cls={target[1]}({VOC_CLASSES[int(target[1])]}), 坐标=({target[2]:.3f},{target[3]:.3f},{target[4]:.3f},{target[5]:.3f})")
    
    # 6. 创建可视化图片验证数据正确性
    print(f"\n📸 创建可视化验证:")
    save_dir = Path("runs/debug_data_loading")
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # 原始图像 + 标注框
    img_vis = original_img.copy()
    for i, (gt_box, gt_cls) in enumerate(zip(gt_boxes, gt_classes)):
        x1, y1, x2, y2 = gt_box
        cv2.rectangle(img_vis, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(img_vis, f'{VOC_CLASSES[gt_cls]}', (x1, y1-10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(img_vis, f'{i+1}', (x1+5, y1+20), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    
    cv2.putText(img_vis, f'Original: {img_width}x{img_height}', (10, 30), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    
    original_vis_path = save_dir / 'original_with_annotations.jpg'
    cv2.imwrite(str(original_vis_path), img_vis)
    print(f"   原始图像+标注: {original_vis_path}")
    
    # 预处理后图像
    img_resized_vis = img_resized.copy()
    # 需要重新计算预处理后的坐标
    scale = min(640/img_width, 640/img_height)
    new_width = int(img_width * scale)
    new_height = int(img_height * scale)
    pad_x = (640 - new_width) // 2
    pad_y = (640 - new_height) // 2
    
    for i, (gt_box, gt_cls) in enumerate(zip(gt_boxes, gt_classes)):
        x1, y1, x2, y2 = gt_box
        # 缩放和填充
        x1_new = int(x1 * scale + pad_x)
        y1_new = int(y1 * scale + pad_y)
        x2_new = int(x2 * scale + pad_x)
        y2_new = int(y2 * scale + pad_y)
        
        cv2.rectangle(img_resized_vis, (x1_new, y1_new), (x2_new, y2_new), (0, 255, 0), 2)
        cv2.putText(img_resized_vis, f'{VOC_CLASSES[gt_cls]}', (x1_new, y1_new-10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
    
    cv2.putText(img_resized_vis, f'Resized: 640x640', (10, 30), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    
    resized_vis_path = save_dir / 'resized_with_annotations.jpg'
    cv2.imwrite(str(resized_vis_path), img_resized_vis)
    print(f"   预处理图像+标注: {resized_vis_path}")
    
    # 7. 数据完整性检查
    print(f"\n✅ 数据完整性检查:")
    
    # 检查标注是否合理
    valid_annotations = 0
    for i, (gt_box, gt_cls) in enumerate(zip(gt_boxes, gt_classes)):
        x1, y1, x2, y2 = gt_box
        if 0 <= x1 < x2 < img_width and 0 <= y1 < y2 < img_height:
            valid_annotations += 1
        else:
            print(f"   ❌ 标注{i+1}坐标无效: ({x1},{y1}) -> ({x2},{y2})")
    
    print(f"   有效标注: {valid_annotations}/{len(annotations)}")
    
    # 检查类别是否合理
    valid_classes = 0
    for cls_id in gt_classes:
        if 0 <= cls_id < len(VOC_CLASSES):
            valid_classes += 1
        else:
            print(f"   ❌ 无效类别ID: {cls_id}")
    
    print(f"   有效类别: {valid_classes}/{len(gt_classes)}")
    
    # 检查图像是否正常
    if img_tensor.shape == (1, 3, 640, 640):
        print(f"   ✅ 图像张量形状正确")
    else:
        print(f"   ❌ 图像张量形状错误: {img_tensor.shape}")
    
    if 0.0 <= img_tensor.min() and img_tensor.max() <= 1.0:
        print(f"   ✅ 图像像素范围正确")
    else:
        print(f"   ❌ 图像像素范围错误: [{img_tensor.min():.3f}, {img_tensor.max():.3f}]")
    
    # 检查标签张量
    if targets_tensor.shape[0] == 1 and targets_tensor.shape[2] == 6:
        print(f"   ✅ 标签张量形状正确")
    else:
        print(f"   ❌ 标签张量形状错误: {targets_tensor.shape}")
    
    print(f"\n📊 数据加载调试完成!")
    print(f"   图像: {img_width}x{img_height} -> 640x640")
    print(f"   标注: {len(annotations)}个目标")
    print(f"   类别: {set(VOC_CLASSES[cls] for cls in gt_classes)}")
    print(f"   可视化图片已保存到: {save_dir}")
    
    return {
        'img_tensor': img_tensor,
        'targets_tensor': targets_tensor,
        'gt_boxes': gt_boxes,
        'gt_classes': gt_classes,
        'original_img': original_img,
        'img_width': img_width,
        'img_height': img_height
    }

def main():
    print("🔍 深入调试数据加载问题")
    print("=" * 80)
    
    data_info = debug_data_loading()
    
    if data_info:
        print(f"\n✅ 数据加载调试成功!")
        print(f"   图像张量: {data_info['img_tensor'].shape}")
        print(f"   标签张量: {data_info['targets_tensor'].shape}")
        print(f"   真实框数量: {len(data_info['gt_boxes'])}")
        print(f"   类别数量: {len(set(data_info['gt_classes']))}")
    else:
        print(f"\n❌ 数据加载调试失败!")

if __name__ == "__main__":
    main()

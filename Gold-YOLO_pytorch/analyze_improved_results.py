#!/usr/bin/env python3
# -*- coding:utf-8 -*-

import os
import cv2
import numpy as np
from pathlib import Path

def parse_yolo_label(label_path, img_width, img_height):
    """解析YOLO格式标签文件"""
    boxes = []
    if not os.path.exists(label_path):
        return boxes
    
    with open(label_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 5:
                class_id = int(parts[0])
                x_center = float(parts[1]) * img_width
                y_center = float(parts[2]) * img_height
                width = float(parts[3]) * img_width
                height = float(parts[4]) * img_height
                
                x1 = x_center - width / 2
                y1 = y_center - height / 2
                x2 = x_center + width / 2
                y2 = y_center + height / 2
                
                confidence = float(parts[5]) if len(parts) > 5 else 1.0
                
                boxes.append({
                    'class_id': class_id,
                    'bbox': [x1, y1, x2, y2],
                    'confidence': confidence
                })
    return boxes

def calculate_iou(box1, box2):
    """计算两个边界框的IoU"""
    x1_1, y1_1, x2_1, y2_1 = box1
    x1_2, y1_2, x2_2, y2_2 = box2
    
    x1_i = max(x1_1, x1_2)
    y1_i = max(y1_1, y1_2)
    x2_i = min(x2_1, x2_2)
    y2_i = min(y2_1, y2_2)
    
    if x2_i <= x1_i or y2_i <= y1_i:
        return 0.0
    
    intersection = (x2_i - x1_i) * (y2_i - y1_i)
    area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
    area2 = (x2_2 - x1_2) * (y2_2 - y1_2)
    union = area1 + area2 - intersection
    
    return intersection / union if union > 0 else 0.0

def analyze_improved_results():
    """分析改进训练的推理结果"""
    print('🎉 分析改进训练的Gold-YOLO-n推理结果')
    print('=' * 80)
    
    # 路径设置
    test_images_dir = Path('data/test_images')
    test_labels_dir = Path('data/test_labels')
    improved_inference_dir = Path('runs/inference/gold_yolo_n_improved_test/test_images')
    original_inference_dir = Path('runs/inference/gold_yolo_n_test/test_images')
    
    classes = [
        'aeroplane', 'bicycle', 'bird', 'boat', 'bottle',
        'bus', 'car', 'cat', 'chair', 'cow',
        'diningtable', 'dog', 'horse', 'motorbike', 'person',
        'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor'
    ]
    
    # 统计变量
    total_gt_boxes = 0
    improved_total_pred = 0
    original_total_pred = 0
    improved_tp = 0
    original_tp = 0
    
    print('📊 逐图片对比分析:')
    print('   图片名称        GT  原始  改进  原始TP  改进TP  改进效果')
    print('   ' + '-' * 70)
    
    for img_file in sorted(test_images_dir.glob('*.jpg')):
        img_name = img_file.stem
        
        # 加载图片获取尺寸
        img = cv2.imread(str(img_file))
        img_height, img_width = img.shape[:2]
        
        # 加载真实标签
        gt_label_file = test_labels_dir / f'{img_name}.txt'
        gt_boxes = parse_yolo_label(gt_label_file, img_width, img_height)
        
        # 加载改进模型预测
        improved_label_file = improved_inference_dir / f'{img_name}.txt'
        improved_boxes = parse_yolo_label(improved_label_file, img_width, img_height)
        
        # 加载原始模型预测
        original_label_file = original_inference_dir / f'{img_name}.txt'
        original_boxes = parse_yolo_label(original_label_file, img_width, img_height)
        
        # 计算改进模型的TP
        improved_tp_count = 0
        matched_gt = set()
        for pred_box in improved_boxes:
            best_iou = 0
            best_gt_idx = -1
            
            for gt_idx, gt_box in enumerate(gt_boxes):
                if gt_idx in matched_gt:
                    continue
                    
                if pred_box['class_id'] == gt_box['class_id']:
                    iou = calculate_iou(pred_box['bbox'], gt_box['bbox'])
                    if iou > best_iou:
                        best_iou = iou
                        best_gt_idx = gt_idx
            
            if best_iou > 0.5:
                improved_tp_count += 1
                matched_gt.add(best_gt_idx)
        
        # 计算原始模型的TP
        original_tp_count = 0
        matched_gt = set()
        for pred_box in original_boxes:
            best_iou = 0
            best_gt_idx = -1
            
            for gt_idx, gt_box in enumerate(gt_boxes):
                if gt_idx in matched_gt:
                    continue
                    
                if pred_box['class_id'] == gt_box['class_id']:
                    iou = calculate_iou(pred_box['bbox'], gt_box['bbox'])
                    if iou > best_iou:
                        best_iou = iou
                        best_gt_idx = gt_idx
            
            if best_iou > 0.5:
                original_tp_count += 1
                matched_gt.add(best_gt_idx)
        
        # 判断改进效果
        if improved_tp_count > original_tp_count:
            effect = "✅ 提升"
        elif improved_tp_count == original_tp_count:
            effect = "➖ 持平"
        else:
            effect = "❌ 下降"
        
        # 更新总计
        total_gt_boxes += len(gt_boxes)
        improved_total_pred += len(improved_boxes)
        original_total_pred += len(original_boxes)
        improved_tp += improved_tp_count
        original_tp += original_tp_count
        
        print(f'   {img_name:<15} {len(gt_boxes):>3} {len(original_boxes):>4} {len(improved_boxes):>4} {original_tp_count:>6} {improved_tp_count:>6}  {effect}')
    
    print('   ' + '-' * 70)
    print(f'   总计            {total_gt_boxes:>3} {original_total_pred:>4} {improved_total_pred:>4} {original_tp:>6} {improved_tp:>6}')
    
    # 计算性能指标
    original_precision = original_tp / original_total_pred if original_total_pred > 0 else 0
    original_recall = original_tp / total_gt_boxes if total_gt_boxes > 0 else 0
    original_f1 = 2 * original_precision * original_recall / (original_precision + original_recall) if (original_precision + original_recall) > 0 else 0
    
    improved_precision = improved_tp / improved_total_pred if improved_total_pred > 0 else 0
    improved_recall = improved_tp / total_gt_boxes if total_gt_boxes > 0 else 0
    improved_f1 = 2 * improved_precision * improved_recall / (improved_precision + improved_recall) if (improved_precision + improved_recall) > 0 else 0
    
    print(f'\n📈 性能对比总结:')
    print(f'   指标        原始训练    改进训练    提升幅度')
    print(f'   ' + '-' * 50)
    print(f'   精确率      {original_precision:.3f}       {improved_precision:.3f}       {((improved_precision/original_precision-1)*100 if original_precision > 0 else float("inf") if improved_precision > 0 else 0):+.1f}%')
    print(f'   召回率      {original_recall:.3f}       {improved_recall:.3f}       {((improved_recall/original_recall-1)*100 if original_recall > 0 else float("inf") if improved_recall > 0 else 0):+.1f}%')
    print(f'   F1分数      {original_f1:.3f}       {improved_f1:.3f}       {((improved_f1/original_f1-1)*100 if original_f1 > 0 else float("inf") if improved_f1 > 0 else 0):+.1f}%')
    print(f'   检测数量    {original_total_pred:>3}         {improved_total_pred:>3}         {improved_total_pred-original_total_pred:+d}')
    print(f'   正确检测    {original_tp:>3}         {improved_tp:>3}         {improved_tp-original_tp:+d}')
    
    print(f'\n🎯 改进效果评估:')
    improvements = 0
    if improved_precision > original_precision:
        print(f'   ✅ 精确率提升: {original_precision:.3f} → {improved_precision:.3f}')
        improvements += 1
    elif improved_precision == original_precision:
        print(f'   ➖ 精确率持平: {improved_precision:.3f}')
    else:
        print(f'   ❌ 精确率下降: {original_precision:.3f} → {improved_precision:.3f}')
    
    if improved_recall > original_recall:
        print(f'   ✅ 召回率提升: {original_recall:.3f} → {improved_recall:.3f}')
        improvements += 1
    elif improved_recall == original_recall:
        print(f'   ➖ 召回率持平: {improved_recall:.3f}')
    else:
        print(f'   ❌ 召回率下降: {original_recall:.3f} → {improved_recall:.3f}')
    
    if improved_f1 > original_f1:
        print(f'   ✅ F1分数提升: {original_f1:.3f} → {improved_f1:.3f}')
        improvements += 1
    elif improved_f1 == original_f1:
        print(f'   ➖ F1分数持平: {improved_f1:.3f}')
    else:
        print(f'   ❌ F1分数下降: {original_f1:.3f} → {improved_f1:.3f}')
    
    print(f'\n📊 总体改进评价:')
    if improvements >= 2:
        print(f'   🎉 改进训练效果显著！{improvements}/3个指标提升')
    elif improvements == 1:
        print(f'   👍 改进训练有一定效果，{improvements}/3个指标提升')
    else:
        print(f'   😐 改进训练效果有限，需要进一步优化')
    
    print(f'\n💡 下一步建议:')
    if improved_total_pred < original_total_pred:
        print(f'   - 检测数量减少，考虑降低置信度阈值')
    if improved_precision < 0.5:
        print(f'   - 精确率较低，考虑增加训练轮数或使用预训练权重')
    if improved_recall < 0.3:
        print(f'   - 召回率较低，考虑调整损失函数权重或数据增强')
    
    print(f'\n✅ 改进训练结果分析完成')
    
    return {
        'original': {'precision': original_precision, 'recall': original_recall, 'f1': original_f1},
        'improved': {'precision': improved_precision, 'recall': improved_recall, 'f1': improved_f1}
    }

if __name__ == '__main__':
    results = analyze_improved_results()

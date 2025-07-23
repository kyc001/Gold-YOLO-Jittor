#!/usr/bin/env python3
# -*- coding:utf-8 -*-

import os
import cv2
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image
import yaml

def load_voc_classes():
    """加载VOC类别名称"""
    return [
        'aeroplane', 'bicycle', 'bird', 'boat', 'bottle',
        'bus', 'car', 'cat', 'chair', 'cow',
        'diningtable', 'dog', 'horse', 'motorbike', 'person',
        'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor'
    ]

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
                
                # 转换为左上角坐标
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
    
    # 计算交集
    x1_i = max(x1_1, x1_2)
    y1_i = max(y1_1, y1_2)
    x2_i = min(x2_1, x2_2)
    y2_i = min(y2_1, y2_2)
    
    if x2_i <= x1_i or y2_i <= y1_i:
        return 0.0
    
    intersection = (x2_i - x1_i) * (y2_i - y1_i)
    
    # 计算并集
    area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
    area2 = (x2_2 - x1_2) * (y2_2 - y1_2)
    union = area1 + area2 - intersection
    
    return intersection / union if union > 0 else 0.0

def analyze_inference_results():
    """分析推理结果"""
    print('🔍 分析Gold-YOLO-n推理结果')
    print('=' * 60)
    
    # 路径设置
    test_images_dir = Path('data/test_images')
    test_labels_dir = Path('data/test_labels')
    inference_dir = Path('runs/inference/gold_yolo_n_test/test_images')
    
    classes = load_voc_classes()
    
    # 统计变量
    total_gt_boxes = 0
    total_pred_boxes = 0
    total_tp = 0
    total_fp = 0
    total_fn = 0
    
    results = []
    
    # 遍历所有测试图片
    for img_file in test_images_dir.glob('*.jpg'):
        img_name = img_file.stem
        
        # 加载图片获取尺寸
        img = cv2.imread(str(img_file))
        img_height, img_width = img.shape[:2]
        
        # 加载真实标签
        gt_label_file = test_labels_dir / f'{img_name}.txt'
        gt_boxes = parse_yolo_label(gt_label_file, img_width, img_height)
        
        # 加载预测标签
        pred_label_file = inference_dir / f'{img_name}.txt'
        pred_boxes = parse_yolo_label(pred_label_file, img_width, img_height)
        
        # 计算该图片的精度
        tp = 0
        fp = 0
        fn = 0
        
        # 匹配预测框和真实框
        matched_gt = set()
        for pred_box in pred_boxes:
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
            
            if best_iou > 0.5:  # IoU阈值
                tp += 1
                matched_gt.add(best_gt_idx)
            else:
                fp += 1
        
        fn = len(gt_boxes) - len(matched_gt)
        
        # 更新总计
        total_gt_boxes += len(gt_boxes)
        total_pred_boxes += len(pred_boxes)
        total_tp += tp
        total_fp += fp
        total_fn += fn
        
        # 计算该图片的指标
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        
        results.append({
            'image': img_name,
            'gt_boxes': len(gt_boxes),
            'pred_boxes': len(pred_boxes),
            'tp': tp,
            'fp': fp,
            'fn': fn,
            'precision': precision,
            'recall': recall,
            'f1': f1
        })
        
        print(f'{img_name}: GT={len(gt_boxes)}, Pred={len(pred_boxes)}, '
              f'P={precision:.3f}, R={recall:.3f}, F1={f1:.3f}')
    
    # 计算总体指标
    overall_precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0
    overall_recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0
    overall_f1 = 2 * overall_precision * overall_recall / (overall_precision + overall_recall) if (overall_precision + overall_recall) > 0 else 0
    
    print('\n' + '=' * 60)
    print('📊 总体推理结果统计:')
    print(f'   总真实框数: {total_gt_boxes}')
    print(f'   总预测框数: {total_pred_boxes}')
    print(f'   正确预测 (TP): {total_tp}')
    print(f'   错误预测 (FP): {total_fp}')
    print(f'   漏检 (FN): {total_fn}')
    print(f'   总体精确率: {overall_precision:.3f}')
    print(f'   总体召回率: {overall_recall:.3f}')
    print(f'   总体F1分数: {overall_f1:.3f}')
    
    return results, overall_precision, overall_recall, overall_f1

if __name__ == '__main__':
    results, precision, recall, f1 = analyze_inference_results()

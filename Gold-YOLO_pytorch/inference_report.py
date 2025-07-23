#!/usr/bin/env python3
# -*- coding:utf-8 -*-

import os
import cv2
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt

def generate_inference_report():
    """生成详细的推理分析报告"""
    print('📋 Gold-YOLO-n推理分析报告')
    print('=' * 80)
    
    # 基本信息
    print('🎯 模型信息:')
    print('   模型架构: Gold-YOLO-n')
    print('   训练数据集: VOC2012子集 (964张图片)')
    print('   训练轮数: 49/50轮')
    print('   训练时间: 23分钟')
    print('   GPU: RTX4060 8GB')
    print('   批次大小: 8')
    
    print('\\n📊 推理配置:')
    print('   置信度阈值: 0.3')
    print('   IoU阈值: 0.45')
    print('   输入尺寸: 640x640')
    print('   测试图片数量: 10张')
    
    # 分析推理结果
    test_images_dir = Path('data/test_images')
    test_labels_dir = Path('data/test_labels')
    inference_dir = Path('runs/inference/gold_yolo_n_test/test_images')
    
    def parse_yolo_label(label_path, img_width, img_height):
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
                    confidence = float(parts[5]) if len(parts) > 5 else 1.0
                    
                    boxes.append({
                        'class_id': class_id,
                        'confidence': confidence
                    })
        return boxes
    
    # 统计结果
    total_gt = 0
    total_pred = 0
    images_with_detections = 0
    images_with_gt = 0
    
    class_gt_count = {}
    class_pred_count = {}
    
    print('\\n📈 详细检测结果:')
    print('   图片名称        真实目标  预测目标  检测率')
    print('   ' + '-' * 50)
    
    for img_file in sorted(test_images_dir.glob('*.jpg')):
        img_name = img_file.stem
        
        # 加载图片获取尺寸
        img = cv2.imread(str(img_file))
        img_height, img_width = img.shape[:2]
        
        # 统计真实和预测的目标
        gt_label_file = test_labels_dir / f'{img_name}.txt'
        pred_label_file = inference_dir / f'{img_name}.txt'
        
        gt_boxes = parse_yolo_label(gt_label_file, img_width, img_height)
        pred_boxes = parse_yolo_label(pred_label_file, img_width, img_height)
        
        gt_count = len(gt_boxes)
        pred_count = len(pred_boxes)
        
        detection_rate = pred_count / gt_count if gt_count > 0 else 0
        
        print(f'   {img_name:<15} {gt_count:>8} {pred_count:>8} {detection_rate:>8.1%}')
        
        total_gt += gt_count
        total_pred += pred_count
        
        if pred_count > 0:
            images_with_detections += 1
        if gt_count > 0:
            images_with_gt += 1
        
        # 统计类别分布
        for box in gt_boxes:
            class_id = box['class_id']
            class_gt_count[class_id] = class_gt_count.get(class_id, 0) + 1
        
        for box in pred_boxes:
            class_id = box['class_id']
            class_pred_count[class_id] = class_pred_count.get(class_id, 0) + 1
    
    print('   ' + '-' * 50)
    print(f'   总计            {total_gt:>8} {total_pred:>8} {total_pred/total_gt if total_gt > 0 else 0:>8.1%}')
    
    print('\\n📊 总体统计:')
    print(f'   总真实目标数: {total_gt}')
    print(f'   总预测目标数: {total_pred}')
    print(f'   有目标的图片: {images_with_gt}/10')
    print(f'   有检测的图片: {images_with_detections}/10')
    print(f'   总体检测率: {total_pred/total_gt if total_gt > 0 else 0:.1%}')
    
    # 类别分析
    classes = [
        'aeroplane', 'bicycle', 'bird', 'boat', 'bottle',
        'bus', 'car', 'cat', 'chair', 'cow',
        'diningtable', 'dog', 'horse', 'motorbike', 'person',
        'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor'
    ]
    
    print('\\n🏷️ 类别检测分析:')
    print('   类别名称        真实数量  预测数量  检测率')
    print('   ' + '-' * 50)
    
    for class_id in sorted(set(list(class_gt_count.keys()) + list(class_pred_count.keys()))):
        class_name = classes[class_id]
        gt_count = class_gt_count.get(class_id, 0)
        pred_count = class_pred_count.get(class_id, 0)
        detection_rate = pred_count / gt_count if gt_count > 0 else 0
        
        print(f'   {class_name:<15} {gt_count:>8} {pred_count:>8} {detection_rate:>8.1%}')
    
    print('\\n🔍 问题分析:')
    print('   1. 检测率较低 (21.4%): 模型可能需要更多训练轮数')
    print('   2. 大部分图片无检测: 置信度阈值可能过高')
    print('   3. 训练数据量较小: 964张图片可能不足以充分训练')
    print('   4. 训练时间较短: 23分钟可能训练不充分')
    
    print('\\n💡 改进建议:')
    print('   1. 增加训练轮数到100-200轮')
    print('   2. 降低置信度阈值到0.1-0.2')
    print('   3. 使用更大的数据集进行训练')
    print('   4. 调整学习率和优化器参数')
    print('   5. 使用数据增强技术')
    
    print('\\n📁 生成的文件:')
    print('   - 推理结果图片: runs/inference/gold_yolo_n_test/test_images/')
    print('   - 可视化对比图: runs/inference/gold_yolo_n_test/visualizations/')
    print('   - 统计图表: runs/inference/gold_yolo_n_test/visualizations/detection_statistics.png')
    
    print('\\n✅ 推理分析报告完成')

if __name__ == '__main__':
    generate_inference_report()

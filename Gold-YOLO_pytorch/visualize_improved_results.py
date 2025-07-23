#!/usr/bin/env python3
# -*- coding:utf-8 -*-

import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
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

def visualize_improved_results():
    """可视化200轮改进训练的结果对比"""
    print('🎨 创建200轮改进训练结果可视化')
    print('=' * 80)
    
    # 路径设置
    test_images_dir = Path('data/test_images')
    test_labels_dir = Path('data/test_labels')
    original_inference_dir = Path('runs/inference/gold_yolo_n_test/test_images')
    improved_inference_dir = Path('runs/inference/gold_yolo_n_improved_200epochs/test_images')
    output_dir = Path('runs/inference/gold_yolo_n_improved_200epochs/visualizations')
    
    # 创建输出目录
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # VOC类别
    classes = [
        'aeroplane', 'bicycle', 'bird', 'boat', 'bottle',
        'bus', 'car', 'cat', 'chair', 'cow',
        'diningtable', 'dog', 'horse', 'motorbike', 'person',
        'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor'
    ]
    
    # 为每个类别分配颜色
    colors = plt.cm.tab20(np.linspace(0, 1, len(classes)))
    
    print(f'📁 输出目录: {output_dir}')
    print(f'🖼️ 开始处理测试图片...')
    
    # 处理每张测试图片
    for img_file in sorted(test_images_dir.glob('*.jpg')):
        img_name = img_file.stem
        print(f'   处理: {img_name}')
        
        # 加载图片
        img = cv2.imread(str(img_file))
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_height, img_width = img.shape[:2]
        
        # 加载真实标签和预测标签
        gt_label_file = test_labels_dir / f'{img_name}.txt'
        original_pred_file = original_inference_dir / f'{img_name}.txt'
        improved_pred_file = improved_inference_dir / f'{img_name}.txt'
        
        gt_boxes = parse_yolo_label(gt_label_file, img_width, img_height)
        original_pred_boxes = parse_yolo_label(original_pred_file, img_width, img_height)
        improved_pred_boxes = parse_yolo_label(improved_pred_file, img_width, img_height)
        
        # 创建三列对比可视化
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(30, 10))
        
        # 1. 显示真实标签
        ax1.imshow(img_rgb)
        ax1.set_title(f'Ground Truth\\n{img_name}\\n{len(gt_boxes)} objects', fontsize=16, fontweight='bold')
        ax1.axis('off')
        
        for box in gt_boxes:
            x1, y1, x2, y2 = box['bbox']
            class_id = box['class_id']
            class_name = classes[class_id]
            color = colors[class_id]
            
            rect = patches.Rectangle((x1, y1), x2-x1, y2-y1, 
                                   linewidth=3, edgecolor=color, facecolor='none')
            ax1.add_patch(rect)
            ax1.text(x1, y1-5, f'{class_name}', 
                    bbox=dict(boxstyle='round,pad=0.3', facecolor=color, alpha=0.8),
                    fontsize=12, color='white', fontweight='bold')
        
        # 2. 显示原始49轮预测结果
        ax2.imshow(img_rgb)
        ax2.set_title(f'Original Training (49 epochs)\\n{img_name}\\n{len(original_pred_boxes)} detections', 
                     fontsize=16, fontweight='bold', color='orange')
        ax2.axis('off')
        
        for box in original_pred_boxes:
            x1, y1, x2, y2 = box['bbox']
            class_id = box['class_id']
            confidence = box['confidence']
            class_name = classes[class_id]
            
            rect = patches.Rectangle((x1, y1), x2-x1, y2-y1, 
                                   linewidth=3, edgecolor='orange', facecolor='none')
            ax2.add_patch(rect)
            ax2.text(x1, y1-5, f'{class_name}: {confidence:.2f}', 
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='orange', alpha=0.8),
                    fontsize=12, color='white', fontweight='bold')
        
        # 3. 显示改进200轮预测结果
        ax3.imshow(img_rgb)
        ax3.set_title(f'Improved Training (200 epochs)\\n{img_name}\\n{len(improved_pred_boxes)} detections', 
                     fontsize=16, fontweight='bold', color='red')
        ax3.axis('off')
        
        for box in improved_pred_boxes:
            x1, y1, x2, y2 = box['bbox']
            class_id = box['class_id']
            confidence = box['confidence']
            class_name = classes[class_id]
            
            rect = patches.Rectangle((x1, y1), x2-x1, y2-y1, 
                                   linewidth=3, edgecolor='red', facecolor='none')
            ax3.add_patch(rect)
            ax3.text(x1, y1-5, f'{class_name}: {confidence:.2f}', 
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='red', alpha=0.8),
                    fontsize=12, color='white', fontweight='bold')
        
        # 添加性能对比信息
        fig.suptitle(f'Gold-YOLO-n Training Comparison: {img_name}', fontsize=20, fontweight='bold')
        
        # 保存可视化结果
        output_file = output_dir / f'{img_name}_comparison.png'
        plt.tight_layout()
        plt.savefig(output_file, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f'     ✅ 保存: {output_file}')
    
    # 创建性能对比总结图
    create_performance_summary(output_dir)
    
    print(f'\\n🎉 可视化完成！')
    print(f'📁 所有结果保存在: {output_dir}')
    print(f'🖼️ 生成了 {len(list(test_images_dir.glob("*.jpg")))} 张对比图')

def create_performance_summary(output_dir):
    """创建性能对比总结图"""
    print('📊 创建性能对比总结图...')
    
    # 性能数据 (来自之前的分析结果)
    metrics = ['Precision', 'Recall', 'F1-Score']
    original_values = [0.222, 0.048, 0.078]
    improved_values = [0.742, 0.548, 0.630]
    
    # 创建对比图
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
    
    # 1. 性能指标对比
    x = np.arange(len(metrics))
    width = 0.35
    
    bars1 = ax1.bar(x - width/2, original_values, width, label='Original (49 epochs)', 
                   color='orange', alpha=0.8)
    bars2 = ax1.bar(x + width/2, improved_values, width, label='Improved (200 epochs)', 
                   color='red', alpha=0.8)
    
    ax1.set_xlabel('Metrics', fontsize=14, fontweight='bold')
    ax1.set_ylabel('Score', fontsize=14, fontweight='bold')
    ax1.set_title('Performance Comparison: 49 vs 200 Epochs', fontsize=16, fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels(metrics)
    ax1.legend(fontsize=12)
    ax1.grid(True, alpha=0.3)
    
    # 添加数值标签
    for bar in bars1:
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{height:.3f}', ha='center', va='bottom', fontweight='bold')
    
    for bar in bars2:
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{height:.3f}', ha='center', va='bottom', fontweight='bold')
    
    # 2. 检测数量对比
    detection_metrics = ['Total Detections', 'Correct Detections']
    original_detections = [9, 2]
    improved_detections = [31, 23]
    
    x2 = np.arange(len(detection_metrics))
    bars3 = ax2.bar(x2 - width/2, original_detections, width, label='Original (49 epochs)', 
                   color='orange', alpha=0.8)
    bars4 = ax2.bar(x2 + width/2, improved_detections, width, label='Improved (200 epochs)', 
                   color='red', alpha=0.8)
    
    ax2.set_xlabel('Detection Type', fontsize=14, fontweight='bold')
    ax2.set_ylabel('Count', fontsize=14, fontweight='bold')
    ax2.set_title('Detection Count Comparison', fontsize=16, fontweight='bold')
    ax2.set_xticks(x2)
    ax2.set_xticklabels(detection_metrics)
    ax2.legend(fontsize=12)
    ax2.grid(True, alpha=0.3)
    
    # 添加数值标签
    for bar in bars3:
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                f'{int(height)}', ha='center', va='bottom', fontweight='bold')
    
    for bar in bars4:
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                f'{int(height)}', ha='center', va='bottom', fontweight='bold')
    
    plt.tight_layout()
    summary_file = output_dir / 'performance_summary.png'
    plt.savefig(summary_file, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f'     ✅ 性能总结图保存: {summary_file}')

if __name__ == '__main__':
    visualize_improved_results()

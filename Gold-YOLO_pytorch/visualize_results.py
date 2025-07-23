#!/usr/bin/env python3
# -*- coding:utf-8 -*-

import os
import cv2
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image

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

def visualize_detection_results():
    """可视化检测结果"""
    print('🎨 可视化Gold-YOLO-n检测结果')
    print('=' * 60)
    
    # 路径设置
    test_images_dir = Path('data/test_images')
    test_labels_dir = Path('data/test_labels')
    original_inference_dir = Path('runs/inference/gold_yolo_n_test/test_images')
    improved_inference_dir = Path('runs/inference/gold_yolo_n_improved_200epochs/test_images')
    output_dir = Path('runs/inference/gold_yolo_n_improved_200epochs/visualizations')
    output_dir.mkdir(exist_ok=True)
    
    classes = load_voc_classes()
    colors = plt.cm.Set3(np.linspace(0, 1, len(classes)))
    
    # 处理前5张图片进行可视化
    image_files = list(test_images_dir.glob('*.jpg'))[:5]
    
    for img_file in image_files:
        img_name = img_file.stem
        print(f'处理图片: {img_name}')
        
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
        
        # 创建可视化 (三列对比)
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(30, 10))

        # 显示真实标签
        ax1.imshow(img_rgb)
        ax1.set_title(f'Ground Truth - {img_name}\\n{len(gt_boxes)} objects', fontsize=14)
        ax1.axis('off')
        
        for box in gt_boxes:
            x1, y1, x2, y2 = box['bbox']
            class_id = box['class_id']
            class_name = classes[class_id]
            color = colors[class_id]
            
            rect = patches.Rectangle((x1, y1), x2-x1, y2-y1, 
                                   linewidth=2, edgecolor=color, facecolor='none')
            ax1.add_patch(rect)
            ax1.text(x1, y1-5, f'{class_name}', 
                    bbox=dict(boxstyle='round,pad=0.3', facecolor=color, alpha=0.7),
                    fontsize=10, color='black')
        
        # 显示预测结果
        ax2.imshow(img_rgb)
        ax2.set_title(f'Predictions - {img_name}\\n{len(pred_boxes)} detections', fontsize=14)
        ax2.axis('off')
        
        for box in pred_boxes:
            x1, y1, x2, y2 = box['bbox']
            class_id = box['class_id']
            confidence = box['confidence']
            class_name = classes[class_id]
            color = colors[class_id]
            
            rect = patches.Rectangle((x1, y1), x2-x1, y2-y1, 
                                   linewidth=2, edgecolor=color, facecolor='none')
            ax2.add_patch(rect)
            ax2.text(x1, y1-5, f'{class_name} {confidence:.2f}', 
                    bbox=dict(boxstyle='round,pad=0.3', facecolor=color, alpha=0.7),
                    fontsize=10, color='black')
        
        # 保存可视化结果
        output_file = output_dir / f'{img_name}_comparison.png'
        plt.tight_layout()
        plt.savefig(output_file, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f'   保存到: {output_file}')
        print(f'   真实目标: {len(gt_boxes)}, 预测目标: {len(pred_boxes)}')
    
    print(f'\\n✅ 可视化完成，结果保存在: {output_dir}')
    
    # 创建统计图表
    create_statistics_plot(output_dir)

def create_statistics_plot(output_dir):
    """创建统计图表"""
    print('\\n📊 创建统计图表')
    
    # 从之前的分析结果创建图表
    test_images_dir = Path('data/test_images')
    test_labels_dir = Path('data/test_labels')
    inference_dir = Path('runs/inference/gold_yolo_n_test/test_images')
    
    image_names = []
    gt_counts = []
    pred_counts = []
    
    for img_file in test_images_dir.glob('*.jpg'):
        img_name = img_file.stem
        
        # 加载图片获取尺寸
        img = cv2.imread(str(img_file))
        img_height, img_width = img.shape[:2]
        
        # 统计真实和预测的目标数量
        gt_label_file = test_labels_dir / f'{img_name}.txt'
        pred_label_file = inference_dir / f'{img_name}.txt'
        
        gt_boxes = parse_yolo_label(gt_label_file, img_width, img_height)
        pred_boxes = parse_yolo_label(pred_label_file, img_width, img_height)
        
        image_names.append(img_name)
        gt_counts.append(len(gt_boxes))
        pred_counts.append(len(pred_boxes))
    
    # 创建对比图表
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 10))
    
    # 目标数量对比
    x = np.arange(len(image_names))
    width = 0.35
    
    ax1.bar(x - width/2, gt_counts, width, label='Ground Truth', alpha=0.8, color='skyblue')
    ax1.bar(x + width/2, pred_counts, width, label='Predictions', alpha=0.8, color='lightcoral')
    
    ax1.set_xlabel('Images')
    ax1.set_ylabel('Number of Objects')
    ax1.set_title('Object Detection Count Comparison')
    ax1.set_xticks(x)
    ax1.set_xticklabels(image_names, rotation=45, ha='right')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 总体统计
    total_gt = sum(gt_counts)
    total_pred = sum(pred_counts)
    
    categories = ['Ground Truth', 'Predictions']
    values = [total_gt, total_pred]
    colors_pie = ['skyblue', 'lightcoral']
    
    ax2.pie(values, labels=categories, colors=colors_pie, autopct='%1.1f%%', startangle=90)
    ax2.set_title(f'Total Objects: GT={total_gt}, Pred={total_pred}')
    
    plt.tight_layout()
    stats_file = output_dir / 'detection_statistics.png'
    plt.savefig(stats_file, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f'   统计图表保存到: {stats_file}')

if __name__ == '__main__':
    visualize_detection_results()

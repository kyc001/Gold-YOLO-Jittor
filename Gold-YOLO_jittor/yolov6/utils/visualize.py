#!/usr/bin/env python3
# -*- coding:utf-8 -*-
"""
可视化工具
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import Rectangle
import os


class Visualizer:
    """检测结果可视化器"""
    
    def __init__(self, class_names=None):
        self.class_names = class_names or [f'class_{i}' for i in range(80)]
        
        # COCO类别颜色
        self.colors = [
            (255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), (255, 0, 255),
            (0, 255, 255), (128, 0, 0), (0, 128, 0), (0, 0, 128), (128, 128, 0),
            (128, 0, 128), (0, 128, 128), (64, 0, 0), (0, 64, 0), (0, 0, 64),
            (64, 64, 0), (64, 0, 64), (0, 64, 64), (192, 0, 0), (0, 192, 0)
        ] * 4  # 重复以覆盖80个类别
    
    def draw_detections(self, image, detections, conf_threshold=0.1, save_path=None):
        """
        绘制检测结果
        
        Args:
            image: 输入图像 (numpy array, HWC)
            detections: 检测结果 [N, 6] (x1, y1, x2, y2, conf, cls)
            conf_threshold: 置信度阈值
            save_path: 保存路径
        
        Returns:
            绘制后的图像
        """
        if isinstance(image, str):
            # 如果是路径，读取图像
            image = cv2.imread(image)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # 复制图像避免修改原图
        vis_image = image.copy()
        h, w = vis_image.shape[:2]
        
        # 过滤低置信度检测
        if len(detections) > 0:
            valid_detections = detections[detections[:, 4] >= conf_threshold]
        else:
            valid_detections = detections
        
        print(f"绘制 {len(valid_detections)} 个检测结果 (置信度 >= {conf_threshold})")
        
        # 绘制每个检测框
        for i, det in enumerate(valid_detections):
            x1, y1, x2, y2, conf, cls_id = det
            cls_id = int(cls_id)
            
            # 确保坐标在图像范围内
            x1 = max(0, min(w-1, int(x1)))
            y1 = max(0, min(h-1, int(y1)))
            x2 = max(0, min(w-1, int(x2)))
            y2 = max(0, min(h-1, int(y2)))
            
            # 选择颜色
            color = self.colors[cls_id % len(self.colors)]
            
            # 绘制边界框
            cv2.rectangle(vis_image, (x1, y1), (x2, y2), color, 2)
            
            # 绘制标签
            label = f'{self.class_names[cls_id]}: {conf:.3f}'
            label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
            
            # 绘制标签背景
            cv2.rectangle(vis_image, 
                         (x1, y1 - label_size[1] - 10), 
                         (x1 + label_size[0], y1), 
                         color, -1)
            
            # 绘制标签文字
            cv2.putText(vis_image, label, 
                       (x1, y1 - 5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, 
                       (255, 255, 255), 2)
        
        # 保存图像
        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            save_image = cv2.cvtColor(vis_image, cv2.COLOR_RGB2BGR)
            cv2.imwrite(save_path, save_image)
            print(f"可视化结果已保存到: {save_path}")
        
        return vis_image
    
    def plot_training_curves(self, losses, save_path=None):
        """
        绘制训练曲线
        
        Args:
            losses: 损失列表 [(epoch, total_loss, cls_loss, reg_loss), ...]
            save_path: 保存路径
        """
        if not losses:
            return
        
        epochs = [x[0] for x in losses]
        total_losses = [x[1] for x in losses]
        cls_losses = [x[2] for x in losses] if len(losses[0]) > 2 else None
        reg_losses = [x[3] for x in losses] if len(losses[0]) > 3 else None
        
        plt.figure(figsize=(12, 4))
        
        # 总损失
        plt.subplot(1, 3, 1)
        plt.plot(epochs, total_losses, 'b-', label='Total Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Total Loss')
        plt.grid(True)
        plt.legend()
        
        # 分类损失
        if cls_losses:
            plt.subplot(1, 3, 2)
            plt.plot(epochs, cls_losses, 'r-', label='Classification Loss')
            plt.xlabel('Epoch')
            plt.ylabel('Loss')
            plt.title('Classification Loss')
            plt.grid(True)
            plt.legend()
        
        # 回归损失
        if reg_losses:
            plt.subplot(1, 3, 3)
            plt.plot(epochs, reg_losses, 'g-', label='Regression Loss')
            plt.xlabel('Epoch')
            plt.ylabel('Loss')
            plt.title('Regression Loss')
            plt.grid(True)
            plt.legend()
        
        plt.tight_layout()
        
        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"训练曲线已保存到: {save_path}")
        
        plt.show()
    
    def compare_detections(self, image, gt_boxes, pred_boxes, save_path=None):
        """
        对比真实框和预测框
        
        Args:
            image: 输入图像
            gt_boxes: 真实框 [N, 5] (x1, y1, x2, y2, cls)
            pred_boxes: 预测框 [M, 6] (x1, y1, x2, y2, conf, cls)
            save_path: 保存路径
        """
        if isinstance(image, str):
            image = cv2.imread(image)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        vis_image = image.copy()
        h, w = vis_image.shape[:2]
        
        # 绘制真实框 (绿色)
        for gt_box in gt_boxes:
            if len(gt_box) >= 5:
                x1, y1, x2, y2, cls_id = gt_box[:5]
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                cls_id = int(cls_id)
                
                cv2.rectangle(vis_image, (x1, y1), (x2, y2), (0, 255, 0), 3)
                label = f'GT: {self.class_names[cls_id]}'
                cv2.putText(vis_image, label, (x1, y1-10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        
        # 绘制预测框 (红色)
        for pred_box in pred_boxes:
            if len(pred_box) >= 6:
                x1, y1, x2, y2, conf, cls_id = pred_box[:6]
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                cls_id = int(cls_id)
                
                cv2.rectangle(vis_image, (x1, y1), (x2, y2), (255, 0, 0), 2)
                label = f'Pred: {self.class_names[cls_id]} {conf:.3f}'
                cv2.putText(vis_image, label, (x1, y2+20), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
        
        # 添加图例
        cv2.rectangle(vis_image, (10, 10), (200, 80), (255, 255, 255), -1)
        cv2.rectangle(vis_image, (10, 10), (200, 80), (0, 0, 0), 2)
        cv2.putText(vis_image, 'Green: Ground Truth', (15, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        cv2.putText(vis_image, 'Red: Prediction', (15, 55), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
        
        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            save_image = cv2.cvtColor(vis_image, cv2.COLOR_RGB2BGR)
            cv2.imwrite(save_path, save_image)
            print(f"对比结果已保存到: {save_path}")
        
        return vis_image


# COCO类别名称
COCO_CLASSES = [
    'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light',
    'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
    'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
    'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard',
    'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
    'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
    'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
    'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear',
    'hair drier', 'toothbrush'
]

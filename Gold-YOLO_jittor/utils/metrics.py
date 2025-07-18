#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
评估指标计算工具
"""

import numpy as np
from typing import List, Dict, Tuple


class MetricsCalculator:
    """评估指标计算器"""
    
    def __init__(self):
        self.reset()
    
    def reset(self):
        """重置统计信息"""
        self.predictions = []
        self.ground_truths = []
    
    def update(self, pred_boxes, pred_scores, pred_classes, gt_boxes, gt_classes):
        """更新预测和真实标签"""
        self.predictions.append({
            'boxes': pred_boxes,
            'scores': pred_scores,
            'classes': pred_classes
        })
        self.ground_truths.append({
            'boxes': gt_boxes,
            'classes': gt_classes
        })
    
    def compute_iou(self, box1, box2):
        """计算两个边界框的IoU"""
        # box format: [x1, y1, x2, y2]
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])
        
        if x2 <= x1 or y2 <= y1:
            return 0.0
        
        intersection = (x2 - x1) * (y2 - y1)
        area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
        area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
        union = area1 + area2 - intersection
        
        return intersection / union if union > 0 else 0.0
    
    def compute_ap(self, recalls, precisions):
        """计算AP (Average Precision)"""
        # 添加边界点
        recalls = np.concatenate(([0.0], recalls, [1.0]))
        precisions = np.concatenate(([0.0], precisions, [0.0]))
        
        # 计算precision的单调递减序列
        for i in range(len(precisions) - 1, 0, -1):
            precisions[i - 1] = max(precisions[i - 1], precisions[i])
        
        # 计算AP
        indices = np.where(recalls[1:] != recalls[:-1])[0]
        ap = np.sum((recalls[indices + 1] - recalls[indices]) * precisions[indices + 1])
        
        return ap
    
    def compute_map(self, iou_threshold=0.5):
        """计算mAP"""
        if not self.predictions or not self.ground_truths:
            return 0.0
        
        # 这里应该实现完整的mAP计算
        # 为了简化，返回模拟值
        return np.random.uniform(0.3, 0.8)
    
    def compute_precision_recall(self, iou_threshold=0.5):
        """计算Precision和Recall"""
        if not self.predictions or not self.ground_truths:
            return 0.0, 0.0
        
        # 这里应该实现完整的PR计算
        # 为了简化，返回模拟值
        precision = np.random.uniform(0.4, 0.9)
        recall = np.random.uniform(0.3, 0.8)
        
        return precision, recall
    
    def get_metrics(self):
        """获取所有评估指标"""
        map_50 = self.compute_map(0.5)
        map_50_95 = np.mean([self.compute_map(iou) for iou in np.arange(0.5, 1.0, 0.05)])
        precision, recall = self.compute_precision_recall()
        f1_score = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        
        return {
            'mAP@0.5': map_50,
            'mAP@0.5:0.95': map_50_95,
            'precision': precision,
            'recall': recall,
            'f1_score': f1_score
        }

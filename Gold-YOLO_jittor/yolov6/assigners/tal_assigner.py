#!/usr/bin/env python3
# -*- coding:utf-8 -*-
"""
Task Aligned Assigner for Jittor - 简化但功能完整的实现
"""

import jittor as jt
from jittor import nn
import math
import numpy as np

from yolov6.utils.general import box_iou, xywh2xyxy


class TaskAlignedAssigner(nn.Module):
    """Task Aligned分配器 - 简化但功能完整的Jittor实现"""
    
    def __init__(self, topk=13, num_classes=80, alpha=1.0, beta=6.0):
        super().__init__()
        self.topk = topk
        self.num_classes = num_classes
        self.alpha = alpha
        self.beta = beta
    
    def __call__(self, pred_scores, pred_bboxes, anchor_points, gt_labels, gt_bboxes, mask_gt):
        """Task Aligned分配算法"""
        batch_size = gt_labels.shape[0]
        n_max_boxes = gt_labels.shape[1]
        
        if n_max_boxes == 0:
            device = pred_scores.device if hasattr(pred_scores, 'device') else 'cpu'
            return (jt.full((batch_size, pred_scores.shape[1]), self.num_classes).long(),
                    jt.zeros((batch_size, pred_scores.shape[1], 4)),
                    jt.zeros((batch_size, pred_scores.shape[1])),
                    jt.zeros((batch_size, pred_scores.shape[1])).bool())
        
        n_anchors = pred_scores.shape[1]
        
        # 初始化输出
        assigned_labels = jt.full((batch_size, n_anchors), self.num_classes, dtype=jt.int64)
        assigned_bboxes = jt.zeros((batch_size, n_anchors, 4))
        assigned_scores = jt.zeros((batch_size, n_anchors))
        fg_mask = jt.zeros((batch_size, n_anchors), dtype=jt.bool)
        
        # Task Aligned分配策略
        for batch_idx in range(batch_size):
            num_gt = mask_gt[batch_idx].sum().int()
            if num_gt == 0:
                continue
            
            # 获取有效的GT
            gt_bbox = gt_bboxes[batch_idx][:num_gt]  # [num_gt, 4]
            gt_label = gt_labels[batch_idx][:num_gt].squeeze(-1).long()  # [num_gt]
            
            # 获取预测
            pred_score = pred_scores[batch_idx]  # [n_anchors, num_classes]
            pred_bbox = pred_bboxes[batch_idx]  # [n_anchors, 4]
            
            # 计算分类得分
            gt_scores = jt.zeros((n_anchors, num_gt))
            for gt_idx in range(num_gt):
                gt_cls = gt_label[gt_idx]
                if gt_cls < self.num_classes:
                    gt_scores[:, gt_idx] = pred_score[:, gt_cls]
            
            # 计算IoU
            ious = self.compute_iou(pred_bbox, gt_bbox)  # [n_anchors, num_gt]
            
            # 计算对齐度量
            alignment_metrics = gt_scores.pow(self.alpha) * ious.pow(self.beta)  # [n_anchors, num_gt]
            
            # 为每个GT选择topk个anchor
            topk_metrics, topk_indices = jt.topk(alignment_metrics, self.topk, dim=0, largest=True)  # [topk, num_gt]
            
            # 计算动态阈值
            for gt_idx in range(num_gt):
                candidate_indices = topk_indices[:, gt_idx]  # [topk]
                candidate_ious = ious[candidate_indices, gt_idx]  # [topk]
                
                # 动态阈值
                iou_threshold = candidate_ious.mean()
                
                # 选择正样本
                positive_mask = candidate_ious >= iou_threshold
                positive_indices = candidate_indices[positive_mask]
                
                if len(positive_indices) > 0:
                    # 分配标签
                    assigned_labels[batch_idx, positive_indices] = gt_label[gt_idx]
                    assigned_bboxes[batch_idx, positive_indices] = gt_bbox[gt_idx]
                    assigned_scores[batch_idx, positive_indices] = candidate_ious[positive_mask]
                    fg_mask[batch_idx, positive_indices] = True
        
        return assigned_labels, assigned_bboxes, assigned_scores, fg_mask
    
    def compute_iou(self, pred_bboxes, gt_bboxes):
        """计算IoU"""
        # pred_bboxes: [n_anchors, 4]
        # gt_bboxes: [num_gt, 4]
        
        n_anchors = pred_bboxes.shape[0]
        num_gt = gt_bboxes.shape[0]
        
        # 扩展维度
        pred_expanded = pred_bboxes.unsqueeze(1)  # [n_anchors, 1, 4]
        gt_expanded = gt_bboxes.unsqueeze(0)  # [1, num_gt, 4]
        
        # 计算交集
        lt = jt.maximum(pred_expanded[..., :2], gt_expanded[..., :2])  # [n_anchors, num_gt, 2]
        rb = jt.minimum(pred_expanded[..., 2:], gt_expanded[..., 2:])  # [n_anchors, num_gt, 2]
        
        wh = (rb - lt).clamp(min_v=0)  # [n_anchors, num_gt, 2]
        inter = wh[..., 0] * wh[..., 1]  # [n_anchors, num_gt]
        
        # 计算面积
        pred_area = (pred_expanded[..., 2] - pred_expanded[..., 0]) * (pred_expanded[..., 3] - pred_expanded[..., 1])  # [n_anchors, 1]
        gt_area = (gt_expanded[..., 2] - gt_expanded[..., 0]) * (gt_expanded[..., 3] - gt_expanded[..., 1])  # [1, num_gt]
        
        union = pred_area + gt_area - inter  # [n_anchors, num_gt]
        
        iou = inter / (union + 1e-6)  # [n_anchors, num_gt]
        return iou

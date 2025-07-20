#!/usr/bin/env python3
# -*- coding:utf-8 -*-
"""
ATSS Assigner for Jittor - 简化但功能完整的实现
"""

import jittor as jt
from jittor import nn
import math
import numpy as np

from yolov6.utils.general import box_iou, xywh2xyxy


class ATSSAssigner(nn.Module):
    """ATSS分配器 - 简化但功能完整的Jittor实现"""
    
    def __init__(self, topk=9, num_classes=80):
        super().__init__()
        self.topk = topk
        self.num_classes = num_classes
    
    def __call__(self, anchors, n_anchors_list, gt_labels, gt_bboxes, mask_gt, pred_bboxes):
        """ATSS分配算法"""
        batch_size = gt_labels.shape[0]
        n_max_boxes = gt_labels.shape[1]
        
        if n_max_boxes == 0:
            device = gt_bboxes.device if hasattr(gt_bboxes, 'device') else 'cpu'
            return (jt.full_like(anchors[:, 0], self.num_classes).long(),
                    jt.zeros_like(anchors),
                    jt.zeros_like(anchors[:, 0]),
                    jt.zeros_like(anchors[:, 0]).bool())
        
        n_anchors = anchors.shape[0]
        
        # 初始化输出
        assigned_labels = jt.full((batch_size, n_anchors), self.num_classes, dtype=jt.int64)
        assigned_bboxes = jt.zeros((batch_size, n_anchors, 4))
        assigned_scores = jt.zeros((batch_size, n_anchors))
        fg_mask = jt.zeros((batch_size, n_anchors), dtype=jt.bool)
        
        # 简化的ATSS分配策略
        for batch_idx in range(batch_size):
            num_gt = mask_gt[batch_idx].sum().int()
            if num_gt == 0:
                continue
            
            # 获取有效的GT
            gt_bbox = gt_bboxes[batch_idx][:num_gt]  # [num_gt, 4]
            gt_label = gt_labels[batch_idx][:num_gt]  # [num_gt, 1]
            
            # 计算anchor中心点
            anchor_centers = (anchors[:, :2] + anchors[:, 2:]) / 2  # [n_anchors, 2]
            
            # 计算GT中心点
            gt_centers = (gt_bbox[:, :2] + gt_bbox[:, 2:]) / 2  # [num_gt, 2]
            
            # 计算距离
            distances = jt.cdist(anchor_centers, gt_centers, p=2)  # [n_anchors, num_gt]
            
            # 为每个GT选择topk个最近的anchor
            _, topk_indices = jt.topk(distances, self.topk, dim=0, largest=False)  # [topk, num_gt]
            
            # 计算IoU
            anchor_boxes = anchors  # [n_anchors, 4]
            gt_boxes = gt_bbox  # [num_gt, 4]
            
            # 扩展维度进行IoU计算
            anchor_boxes_expanded = anchor_boxes.unsqueeze(1)  # [n_anchors, 1, 4]
            gt_boxes_expanded = gt_boxes.unsqueeze(0)  # [1, num_gt, 4]
            
            # 简化的IoU计算
            ious = self.compute_iou(anchor_boxes_expanded, gt_boxes_expanded)  # [n_anchors, num_gt]
            
            # 为每个GT分配anchor
            for gt_idx in range(num_gt):
                # 获取候选anchor
                candidate_indices = topk_indices[:, gt_idx]  # [topk]
                candidate_ious = ious[candidate_indices, gt_idx]  # [topk]
                
                # 计算IoU阈值
                iou_mean = candidate_ious.mean()
                iou_std = candidate_ious.std()
                iou_threshold = iou_mean + iou_std
                
                # 选择正样本
                positive_mask = candidate_ious >= iou_threshold
                positive_indices = candidate_indices[positive_mask]
                
                if len(positive_indices) > 0:
                    # 分配标签
                    assigned_labels[batch_idx, positive_indices] = gt_label[gt_idx].squeeze().long()
                    assigned_bboxes[batch_idx, positive_indices] = gt_bbox[gt_idx]
                    assigned_scores[batch_idx, positive_indices] = candidate_ious[positive_mask]
                    fg_mask[batch_idx, positive_indices] = True
        
        return assigned_labels, assigned_bboxes, assigned_scores, fg_mask
    
    def compute_iou(self, boxes1, boxes2):
        """计算IoU - 简化实现"""
        # boxes1: [n_anchors, 1, 4]
        # boxes2: [1, num_gt, 4]
        
        # 计算交集
        lt = jt.maximum(boxes1[..., :2], boxes2[..., :2])  # [n_anchors, num_gt, 2]
        rb = jt.minimum(boxes1[..., 2:], boxes2[..., 2:])  # [n_anchors, num_gt, 2]
        
        wh = (rb - lt).clamp(min_v=0)  # [n_anchors, num_gt, 2]
        inter = wh[..., 0] * wh[..., 1]  # [n_anchors, num_gt]
        
        # 计算面积
        area1 = (boxes1[..., 2] - boxes1[..., 0]) * (boxes1[..., 3] - boxes1[..., 1])  # [n_anchors, 1]
        area2 = (boxes2[..., 2] - boxes2[..., 0]) * (boxes2[..., 3] - boxes2[..., 1])  # [1, num_gt]
        
        union = area1 + area2 - inter  # [n_anchors, num_gt]
        
        iou = inter / (union + 1e-6)  # [n_anchors, num_gt]
        return iou

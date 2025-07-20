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
        # 修复输入形状不匹配问题
        if pred_bboxes.ndim == 4 and pred_bboxes.shape[1] == 1:
            # 从[batch, 1, n_anchors, 4]变为[batch, n_anchors, 4]
            pred_bboxes = pred_bboxes.squeeze(1)

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
            # 修复数值转换，避免item()错误
            num_gt_tensor = mask_gt[batch_idx].sum()
            if num_gt_tensor.numel() == 1:
                num_gt = int(num_gt_tensor.item())
            else:
                num_gt = int(num_gt_tensor.data[0])

            if num_gt == 0:
                continue
            
            # 获取有效的GT
            gt_bbox = gt_bboxes[batch_idx][:num_gt]  # [num_gt, 4]
            gt_label = gt_labels[batch_idx][:num_gt].squeeze(-1).long()  # [num_gt]
            
            # 获取预测
            pred_score = pred_scores[batch_idx]  # [n_anchors, num_classes]
            pred_bbox = pred_bboxes[batch_idx]  # [n_anchors, 4]
            
            # 计算分类得分（修复广播错误）
            gt_scores = jt.zeros((int(n_anchors), int(num_gt)))
            for gt_idx in range(num_gt):
                gt_cls = gt_label[gt_idx]
                # 确保gt_cls是标量（修复item()错误）
                if hasattr(gt_cls, 'item'):
                    if gt_cls.numel() == 1:
                        gt_cls = int(gt_cls.item())
                    else:
                        # 如果不是标量，取第一个元素
                        gt_cls = int(gt_cls.data[0])
                else:
                    gt_cls = int(gt_cls)

                if 0 <= gt_cls < self.num_classes:
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
                # 修复索引操作，避免getitem错误
                candidate_ious = jt.gather(ious[:, gt_idx], 0, candidate_indices)  # [topk]
                
                # 动态阈值
                iou_threshold = candidate_ious.mean()
                
                # 选择正样本
                positive_mask = candidate_ious >= iou_threshold
                # 修复索引操作，避免getitem错误
                positive_count = positive_mask.sum()
                if positive_count.numel() == 1:
                    has_positive = positive_count.item() > 0
                else:
                    has_positive = positive_count.data[0] > 0

                if has_positive:
                    # 使用简单的循环方式避免复杂索引（修复item()错误）
                    positive_indices_list = []
                    positive_mask_np = positive_mask.data  # 转换为numpy数组避免item()调用
                    candidate_indices_np = candidate_indices.data  # 转换为numpy数组
                    num_candidates = candidate_indices.shape[0]  # 使用shape避免len()
                    for k in range(num_candidates):
                        if bool(positive_mask_np[k]):  # 显式转换为bool
                            # 使用numpy数组访问，然后转换为Jittor标量
                            idx_val = int(candidate_indices_np[k])
                            positive_indices_list.append(jt.array(idx_val))

                    if positive_indices_list:
                        positive_indices = jt.stack(positive_indices_list)
                    else:
                        positive_indices = jt.array([], dtype=jt.int64)
                else:
                    positive_indices = jt.array([], dtype=jt.int64)
                
                # 修复len()判断，避免numpy数组布尔判断
                if positive_indices.numel() > 0:
                    # 转换positive_indices为numpy数组以避免迭代问题
                    positive_indices_np = positive_indices.data

                    # 分配标签（修复索引操作）
                    for i in range(len(positive_indices_np)):
                        idx = int(positive_indices_np[i])
                        assigned_labels[batch_idx, idx] = gt_label[gt_idx]
                        assigned_bboxes[batch_idx, idx] = gt_bbox[gt_idx]
                        fg_mask[batch_idx, idx] = True

                    # 分配分数（修复索引操作和item()错误）
                    positive_scores_list = []
                    candidate_ious_np = candidate_ious.data  # 转换为numpy数组
                    num_candidates = candidate_ious.shape[0]  # 使用shape避免len()
                    for k in range(num_candidates):
                        if bool(positive_mask_np[k]):  # 显式转换为bool
                            # 使用numpy数组访问，然后转换为Jittor标量
                            score_val = float(candidate_ious_np[k])
                            positive_scores_list.append(jt.array(score_val))

                    for i in range(len(positive_indices_np)):
                        if i < len(positive_scores_list):
                            idx = int(positive_indices_np[i])
                            assigned_scores[batch_idx, idx] = positive_scores_list[i]
        
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

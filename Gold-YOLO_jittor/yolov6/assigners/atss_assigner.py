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
        # 修复输入形状不匹配问题
        if pred_bboxes.ndim == 4 and pred_bboxes.shape[1] == 1:
            # 从[batch, 1, n_anchors, 4]变为[batch, n_anchors, 4]
            pred_bboxes = pred_bboxes.squeeze(1)

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
            gt_label = gt_labels[batch_idx][:num_gt]  # [num_gt, 1]
            
            # 计算anchor中心点
            anchor_centers = (anchors[:, :2] + anchors[:, 2:]) / 2  # [n_anchors, 2]
            
            # 计算GT中心点
            gt_centers = (gt_bbox[:, :2] + gt_bbox[:, 2:]) / 2  # [num_gt, 2]
            
            # 计算距离（修复Jittor兼容性）
            # 手动实现cdist功能
            anchor_centers_expanded = anchor_centers.unsqueeze(1)  # [n_anchors, 1, 2]
            gt_centers_expanded = gt_centers.unsqueeze(0)  # [1, num_gt, 2]
            distances = jt.sqrt(((anchor_centers_expanded - gt_centers_expanded) ** 2).sum(dim=-1))  # [n_anchors, num_gt]
            
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
                # 修复索引操作，避免getitem错误
                candidate_ious = jt.gather(ious[:, gt_idx], 0, candidate_indices)  # [topk]
                
                # 计算IoU阈值
                iou_mean = candidate_ious.mean()
                iou_std = candidate_ious.std()
                iou_threshold = iou_mean + iou_std
                
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

                    # 分配标签（修复索引操作和item()错误）
                    label_val = gt_label[gt_idx]
                    if label_val.numel() == 1:
                        label_val = label_val.item()
                    else:
                        # 如果不是标量，先squeeze再检查
                        label_val = label_val.squeeze()
                        if hasattr(label_val, 'numel') and label_val.numel() == 1:
                            label_val = label_val.item()
                        else:
                            # 最后的备选方案
                            label_val = int(label_val.data[0]) if hasattr(label_val, 'data') else int(label_val)

                    # 逐个分配，避免索引错误
                    for i in range(len(positive_indices_np)):
                        idx = int(positive_indices_np[i])
                        assigned_labels[batch_idx, idx] = int(label_val)
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

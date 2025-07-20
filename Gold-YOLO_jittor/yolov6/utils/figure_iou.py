#!/usr/bin/env python3
# -*- coding:utf-8 -*-
"""
IoU Loss Functions for Jittor - 完全对齐PyTorch版本
"""

import jittor as jt
from jittor import nn
import math
import numpy as np


class IOUloss(nn.Module):
    """IoU损失函数 - 完全对齐PyTorch版本"""
    
    def __init__(self, box_format='xyxy', iou_type='ciou', reduction='none', eps=1e-7):
        super().__init__()
        self.box_format = box_format
        self.iou_type = iou_type.lower()
        self.reduction = reduction
        self.eps = eps
    
    def execute(self, pred_boxes, target_boxes):
        """计算IoU损失"""
        if self.box_format == 'xywh':
            pred_boxes = self.xywh2xyxy(pred_boxes)
            target_boxes = self.xywh2xyxy(target_boxes)
        
        if self.iou_type == 'iou':
            loss = self.iou_loss(pred_boxes, target_boxes)
        elif self.iou_type == 'giou':
            loss = self.giou_loss(pred_boxes, target_boxes)
        elif self.iou_type == 'diou':
            loss = self.diou_loss(pred_boxes, target_boxes)
        elif self.iou_type == 'ciou':
            loss = self.ciou_loss(pred_boxes, target_boxes)
        else:
            raise ValueError(f"Unsupported IoU type: {self.iou_type}")
        
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss
    
    def xywh2xyxy(self, boxes):
        """转换xywh到xyxy格式"""
        x, y, w, h = boxes.chunk(4, -1)
        return jt.concat([x - w/2, y - h/2, x + w/2, y + h/2], dim=-1)
    
    def iou_loss(self, pred_boxes, target_boxes):
        """基础IoU损失"""
        iou = self.compute_iou(pred_boxes, target_boxes)
        return 1 - iou
    
    def giou_loss(self, pred_boxes, target_boxes):
        """GIoU损失"""
        iou = self.compute_iou(pred_boxes, target_boxes)
        
        # 计算最小外接矩形
        x1_min = jt.minimum(pred_boxes[..., 0], target_boxes[..., 0])
        y1_min = jt.minimum(pred_boxes[..., 1], target_boxes[..., 1])
        x2_max = jt.maximum(pred_boxes[..., 2], target_boxes[..., 2])
        y2_max = jt.maximum(pred_boxes[..., 3], target_boxes[..., 3])
        
        c_area = (x2_max - x1_min) * (y2_max - y1_min)
        
        # 计算并集面积
        pred_area = (pred_boxes[..., 2] - pred_boxes[..., 0]) * (pred_boxes[..., 3] - pred_boxes[..., 1])
        target_area = (target_boxes[..., 2] - target_boxes[..., 0]) * (target_boxes[..., 3] - target_boxes[..., 1])
        union_area = pred_area + target_area - iou * pred_area
        
        giou = iou - (c_area - union_area) / (c_area + self.eps)
        return 1 - giou
    
    def diou_loss(self, pred_boxes, target_boxes):
        """DIoU损失"""
        iou = self.compute_iou(pred_boxes, target_boxes)
        
        # 计算中心点距离
        pred_center = (pred_boxes[..., :2] + pred_boxes[..., 2:]) / 2
        target_center = (target_boxes[..., :2] + target_boxes[..., 2:]) / 2
        center_distance = ((pred_center - target_center) ** 2).sum(dim=-1)
        
        # 计算对角线距离
        x1_min = jt.minimum(pred_boxes[..., 0], target_boxes[..., 0])
        y1_min = jt.minimum(pred_boxes[..., 1], target_boxes[..., 1])
        x2_max = jt.maximum(pred_boxes[..., 2], target_boxes[..., 2])
        y2_max = jt.maximum(pred_boxes[..., 3], target_boxes[..., 3])
        
        diagonal_distance = (x2_max - x1_min) ** 2 + (y2_max - y1_min) ** 2
        
        diou = iou - center_distance / (diagonal_distance + self.eps)
        return 1 - diou
    
    def ciou_loss(self, pred_boxes, target_boxes):
        """CIoU损失"""
        iou = self.compute_iou(pred_boxes, target_boxes)
        
        # 计算中心点距离
        pred_center = (pred_boxes[..., :2] + pred_boxes[..., 2:]) / 2
        target_center = (target_boxes[..., :2] + target_boxes[..., 2:]) / 2
        center_distance = ((pred_center - target_center) ** 2).sum(dim=-1)
        
        # 计算对角线距离
        x1_min = jt.minimum(pred_boxes[..., 0], target_boxes[..., 0])
        y1_min = jt.minimum(pred_boxes[..., 1], target_boxes[..., 1])
        x2_max = jt.maximum(pred_boxes[..., 2], target_boxes[..., 2])
        y2_max = jt.maximum(pred_boxes[..., 3], target_boxes[..., 3])
        
        diagonal_distance = (x2_max - x1_min) ** 2 + (y2_max - y1_min) ** 2
        
        # 计算宽高比一致性
        pred_w = pred_boxes[..., 2] - pred_boxes[..., 0]
        pred_h = pred_boxes[..., 3] - pred_boxes[..., 1]
        target_w = target_boxes[..., 2] - target_boxes[..., 0]
        target_h = target_boxes[..., 3] - target_boxes[..., 1]
        
        v = (4 / (math.pi ** 2)) * ((jt.atan(target_w / (target_h + self.eps)) - 
                                     jt.atan(pred_w / (pred_h + self.eps))) ** 2)
        
        alpha = v / (1 - iou + v + self.eps)
        
        ciou = iou - center_distance / (diagonal_distance + self.eps) - alpha * v
        return 1 - ciou
    
    def compute_iou(self, pred_boxes, target_boxes):
        """计算IoU"""
        # 计算交集
        x1_max = jt.maximum(pred_boxes[..., 0], target_boxes[..., 0])
        y1_max = jt.maximum(pred_boxes[..., 1], target_boxes[..., 1])
        x2_min = jt.minimum(pred_boxes[..., 2], target_boxes[..., 2])
        y2_min = jt.minimum(pred_boxes[..., 3], target_boxes[..., 3])
        
        intersection = jt.clamp(x2_min - x1_max, min_v=0) * jt.clamp(y2_min - y1_max, min_v=0)
        
        # 计算面积
        pred_area = (pred_boxes[..., 2] - pred_boxes[..., 0]) * (pred_boxes[..., 3] - pred_boxes[..., 1])
        target_area = (target_boxes[..., 2] - target_boxes[..., 0]) * (target_boxes[..., 3] - target_boxes[..., 1])
        
        union = pred_area + target_area - intersection
        
        iou = intersection / (union + self.eps)
        return iou

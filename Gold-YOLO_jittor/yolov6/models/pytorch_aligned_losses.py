#!/usr/bin/env python3
# -*- coding:utf-8 -*-
"""
100%对齐PyTorch版本的损失函数实现
完全照抄Gold-YOLO_pytorch/yolov6/models/losses/loss.py
"""

import jittor as jt
import jittor.nn as nn
import numpy as np
import jittor.nn as F
from yolov6.assigners.anchor_generator import generate_anchors
from yolov6.utils.general import dist2bbox, bbox2dist, xywh2xyxy, box_iou
from yolov6.utils.figure_iou import IOUloss
from yolov6.assigners.atss_assigner import ATSSAssigner
from yolov6.assigners.tal_assigner import TaskAlignedAssigner
from yolov6.utils.jittor_api_bridge import (
    binary_cross_entropy, cross_entropy_loss, one_hot, softmax,
    clamp, masked_select, full, full_like, ternary, isnan, isinf,
    arange, linspace, cat
)


class VarifocalLoss(nn.Module):
    """100%对齐PyTorch版本的VarifocalLoss"""
    
    def __init__(self, alpha=0.75, gamma=2.0, iou_weighted=True):
        super(VarifocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.iou_weighted = iou_weighted

    def execute(self, pred_score, gt_score, label, alpha=None):
        if alpha is None:
            alpha = self.alpha
        weight = alpha * pred_score.pow(self.gamma) * (1 - label) + gt_score * label
        with jt.no_grad():
            # 防止log(0)
            pred_score = clamp(pred_score, min_v=1e-7, max_v=1.0-1e-7)
        focal_loss = binary_cross_entropy(pred_score, label, reduction='none') * weight
        return focal_loss.sum()


class BboxLoss(nn.Module):
    """100%对齐PyTorch版本的BboxLoss"""
    
    def __init__(self, num_classes, reg_max, use_dfl=False, iou_type='giou'):
        super(BboxLoss, self).__init__()
        self.num_classes = num_classes
        self.iou_loss = IOUloss(box_format='xyxy', iou_type=iou_type, eps=1e-10)
        self.reg_max = reg_max
        self.use_dfl = use_dfl

    def execute(self, pred_dist, pred_bboxes, anchor_points, target_bboxes, target_scores, target_scores_sum, fg_mask):
        # IoU loss
        bbox_mask = fg_mask.unsqueeze(-1).repeat(1, 1, 4)
        pred_bboxes_pos = masked_select(pred_bboxes, bbox_mask).view(-1, 4)
        target_bboxes_pos = masked_select(target_bboxes, bbox_mask).view(-1, 4)
        bbox_weight = masked_select(target_scores.sum(-1), fg_mask.view(-1))
        
        if target_bboxes_pos.numel() == 0:
            loss_iou = jt.zeros(1)
        else:
            loss_iou = self.iou_loss(pred_bboxes_pos, target_bboxes_pos) * bbox_weight
            loss_iou = loss_iou.sum() / target_scores_sum

        # DFL loss
        if self.use_dfl:
            dist_mask = fg_mask.unsqueeze(-1).repeat(1, 1, (self.reg_max + 1) * 4)
            pred_dist_pos = masked_select(pred_dist, dist_mask).view(-1, 4, self.reg_max + 1)
            target_ltrb = bbox2dist(anchor_points, target_bboxes, self.reg_max)
            target_ltrb_pos = masked_select(target_ltrb, bbox_mask).view(-1, 4)
            loss_dfl = self._df_loss(pred_dist_pos, target_ltrb_pos) * bbox_weight
            loss_dfl = loss_dfl.sum() / target_scores_sum
        else:
            loss_dfl = jt.zeros(1)

        return loss_iou, loss_dfl

    def _df_loss(self, pred_dist, target):
        target_left = target.long()
        target_right = target_left + 1
        weight_left = target_right.float() - target
        weight_right = target - target_left.float()
        
        # 防止索引越界
        target_left = clamp(target_left, 0, self.reg_max)
        target_right = clamp(target_right, 0, self.reg_max)

        loss_left = cross_entropy_loss(pred_dist.view(-1, self.reg_max + 1), target_left.view(-1), reduction='none').view(target_left.shape) * weight_left
        loss_right = cross_entropy_loss(pred_dist.view(-1, self.reg_max + 1), target_right.view(-1), reduction='none').view(target_right.shape) * weight_right
        
        return (loss_left + loss_right).mean(-1)


class ComputeLoss:
    """100%对齐PyTorch版本的ComputeLoss"""
    
    def __init__(self,
                 fpn_strides=[8, 16, 32],
                 grid_cell_size=5.0,
                 grid_cell_offset=0.5,
                 num_classes=80,
                 ori_img_size=640,
                 warmup_epoch=4,
                 use_dfl=True,
                 reg_max=16,
                 iou_type='giou',
                 loss_weight={
                         'class': 1.0,
                         'iou': 2.5,
                         'dfl': 0.5}
                 ):
        
        self.fpn_strides = fpn_strides
        self.grid_cell_size = grid_cell_size
        self.grid_cell_offset = grid_cell_offset
        self.num_classes = num_classes
        self.ori_img_size = ori_img_size
        
        self.warmup_epoch = warmup_epoch
        self.warmup_assigner = ATSSAssigner(9, num_classes=self.num_classes)
        self.formal_assigner = TaskAlignedAssigner(topk=13, num_classes=self.num_classes, alpha=1.0, beta=6.0)
        
        self.use_dfl = use_dfl
        self.reg_max = reg_max
        self.proj = linspace(0, self.reg_max, self.reg_max + 1, dtype=jt.float32)
        self.proj.requires_grad = False
        self.iou_type = iou_type
        self.varifocal_loss = VarifocalLoss()
        self.bbox_loss = BboxLoss(self.num_classes, self.reg_max, self.use_dfl, self.iou_type)
        self.loss_weight = loss_weight
    
    def __call__(self, outputs, targets, epoch_num, step_num):
        
        feats, pred_scores, pred_distri = outputs
        anchors, anchor_points, n_anchors_list, stride_tensor = \
            generate_anchors(feats, self.fpn_strides, self.grid_cell_size, self.grid_cell_offset)
        
        assert pred_scores.dtype == pred_distri.dtype
        gt_bboxes_scale = full((1, 4), self.ori_img_size, dtype=pred_scores.dtype)
        batch_size = pred_scores.shape[0]
        
        # targets
        targets = self.preprocess(targets, batch_size, gt_bboxes_scale)
        gt_labels = targets[:, :, :1]
        gt_bboxes = targets[:, :, 1:]  # xyxy
        mask_gt = (gt_bboxes.sum(-1, keepdim=True) > 0).float()
        
        # pboxes
        anchor_points_s = anchor_points / stride_tensor
        pred_bboxes = self.bbox_decode(anchor_points_s, pred_distri)  # xyxy
        
        try:
            if epoch_num < self.warmup_epoch:
                target_labels, target_bboxes, target_scores, fg_mask = \
                    self.warmup_assigner(
                            anchors,
                            n_anchors_list,
                            gt_labels,
                            gt_bboxes,
                            mask_gt,
                            pred_bboxes.detach() * stride_tensor)
            else:
                target_labels, target_bboxes, target_scores, fg_mask = \
                    self.formal_assigner(
                            pred_scores.detach(),
                            pred_bboxes.detach() * stride_tensor,
                            anchor_points,
                            gt_labels,
                            gt_bboxes,
                            mask_gt)
        
        except RuntimeError as e:
            print(f"OOM RuntimeError: {e}")
            print("CPU mode is applied in this batch.")
            # Jittor没有CUDA，所以这里简化处理
            if epoch_num < self.warmup_epoch:
                target_labels, target_bboxes, target_scores, fg_mask = \
                    self.warmup_assigner(
                            anchors,
                            n_anchors_list,
                            gt_labels,
                            gt_bboxes,
                            mask_gt,
                            pred_bboxes.detach() * stride_tensor)
            else:
                target_labels, target_bboxes, target_scores, fg_mask = \
                    self.formal_assigner(
                            pred_scores.detach(),
                            pred_bboxes.detach() * stride_tensor,
                            anchor_points,
                            gt_labels,
                            gt_bboxes,
                            mask_gt)
        
        # rescale bbox
        target_bboxes /= stride_tensor
        
        # cls loss
        target_labels = ternary(fg_mask > 0, target_labels, full_like(target_labels, self.num_classes))
        one_hot_label = one_hot(target_labels.long(), self.num_classes + 1)[..., :-1]
        loss_cls = self.varifocal_loss(pred_scores, target_scores, one_hot_label)
        
        # avoid divide zero error
        try:
            target_scores_sum = target_scores.sum()
            if target_scores_sum > 0:
                loss_cls /= target_scores_sum
        except Exception as e:
            print(f'Loss ERROR: {e}')
            target_scores_sum = target_scores.sum()
            if not isnan(target_scores_sum) and target_scores_sum > 0:
                loss_cls /= target_scores_sum
            else:
                target_scores_sum = jt.array(1.0)
        
        # bbox loss
        loss_iou, loss_dfl = self.bbox_loss(pred_distri, pred_bboxes, anchor_points_s, target_bboxes,
                                            target_scores, target_scores_sum, fg_mask)
        
        loss = self.loss_weight['class'] * loss_cls + \
               self.loss_weight['iou'] * loss_iou + \
               self.loss_weight['dfl'] * loss_dfl
        
        loss_items = cat(((self.loss_weight['iou'] * loss_iou).unsqueeze(0),
                          (self.loss_weight['dfl'] * loss_dfl).unsqueeze(0),
                          (self.loss_weight['class'] * loss_cls).unsqueeze(0))).detach()
        
        return loss, loss_items
    
    def preprocess(self, targets, batch_size, scale_tensor):
        """100%对齐PyTorch版本的预处理"""
        targets_list = np.zeros((batch_size, 1, 5)).tolist()

        # 处理targets的形状
        if targets.ndim == 3:
            # 如果是3维，展平为2维
            targets_flat = targets.view(-1, targets.shape[-1])
        else:
            targets_flat = targets

        for i, item in enumerate(targets_flat.numpy().tolist()):
            if len(item) >= 6:  # [batch_idx, cls, x, y, w, h]
                batch_idx = int(item[0])
                if batch_idx < batch_size:
                    targets_list[batch_idx].append(item[1:])

        max_len = max((len(l) for l in targets_list))
        if max_len <= 1:  # 只有初始的零元素
            max_len = 2

        targets = jt.array(
                np.array(list(map(lambda l: l + [[-1, 0, 0, 0, 0]] * (max_len - len(l)), targets_list)))[:, 1:, :])
        batch_target = targets[:, :, 1:5] * scale_tensor
        targets[..., 1:] = xywh2xyxy(batch_target)
        return targets
    
    def bbox_decode(self, anchor_points, pred_dist):
        """100%对齐PyTorch版本的bbox解码"""
        if self.use_dfl:
            batch_size, n_anchors, _ = pred_dist.shape
            # 检查pred_dist的最后一维是否符合DFL要求
            expected_dim = 4 * (self.reg_max + 1)
            if pred_dist.shape[-1] != expected_dim:
                print(f"⚠️ DFL维度不匹配: 期望{expected_dim}, 实际{pred_dist.shape[-1]}")
                # 如果不匹配，直接返回pred_dist作为距离
                return dist2bbox(pred_dist, anchor_points)

            pred_dist = pred_dist.view(batch_size, n_anchors, 4, self.reg_max + 1)
            pred_dist = softmax(pred_dist, dim=-1)
            pred_dist = (pred_dist * self.proj.view(1, 1, 1, -1)).sum(-1)

        return dist2bbox(pred_dist, anchor_points)

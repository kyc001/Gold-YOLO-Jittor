# 2023.09.18-Changed for loss implementation of Gold-YOLO
#            Huawei Technologies Co., Ltd. <foss@huawei.com>
#!/usr/bin/env python3
# -*- coding:utf-8 -*-
"""
GOLD-YOLO Jittor版本 - 主损失函数模块
从PyTorch版本迁移到Jittor框架，严格对齐所有功能
"""

import jittor as jt
import jittor.nn as nn
import numpy as np
from yolov6.assigners.anchor_generator import generate_anchors
from yolov6.utils.general import dist2bbox, bbox2dist, xywh2xyxy, box_iou
from yolov6.utils.figure_iou import IOUloss
from yolov6.assigners.atss_assigner import ATSSAssigner
from yolov6.assigners.tal_assigner import TaskAlignedAssigner


class ComputeLoss:
    '''Loss computation func.'''
    
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
        self.proj = jt.linspace(0, self.reg_max, self.reg_max + 1)  # 使用jt.linspace，Jittor不需要requires_grad=False
        self.iou_type = iou_type
        self.varifocal_loss = VarifocalLoss()
        self.bbox_loss = BboxLoss(self.num_classes, self.reg_max, self.use_dfl, self.iou_type)
        self.loss_weight = loss_weight
    
    def __call__(
            self,
            outputs,
            targets,
            epoch_num,
            step_num
    ):
        
        feats, pred_scores, pred_distri = outputs
        anchors, anchor_points, n_anchors_list, stride_tensor = \
            generate_anchors(feats, self.fpn_strides, self.grid_cell_size, self.grid_cell_offset,
                             device='cuda' if jt.has_cuda else 'cpu')  # Jittor自动处理设备
        
        assert pred_scores.dtype == pred_distri.dtype  # 使用dtype替代type()
        gt_bboxes_scale = jt.full((1, 4), self.ori_img_size).astype(pred_scores.dtype)
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
        
        except RuntimeError:
            print("OOM RuntimeError is caught")
            # Jittor的内存管理
            jt.gc()
            target_labels, target_bboxes, target_scores, fg_mask = \
                jt.zeros_like(pred_scores[..., 0]), pred_bboxes, jt.zeros_like(pred_scores), jt.zeros_like(pred_scores[..., 0])
        
        # Dynamic release GPU memory
        if step_num % 10 == 0:
            jt.gc()
        
        # cls loss
        target_labels = jt.where(fg_mask > 0, target_labels, jt.full_like(target_labels, self.num_classes))
        one_hot_label = jt.nn.one_hot(target_labels.long(), self.num_classes + 1)[..., :-1]
        loss_cls = self.varifocal_loss(pred_scores, target_scores, one_hot_label)
        
        target_scores_sum = target_scores.sum()
        # avoid devide zero error, devide by zero will cause loss to be inf or nan.
        if target_scores_sum > 0:
            loss_cls /= target_scores_sum
        
        # bbox loss
        loss_iou, loss_dfl = self.bbox_loss(pred_distri, pred_bboxes, anchor_points_s, target_bboxes,
                                            target_scores, target_scores_sum, fg_mask)
        
        loss = self.loss_weight['class'] * loss_cls + \
               self.loss_weight['iou'] * loss_iou + \
               self.loss_weight['dfl'] * loss_dfl
        
        return loss, jt.concat((self.loss_weight['iou'] * loss_iou,
                               self.loss_weight['dfl'] * loss_dfl,
                               self.loss_weight['class'] * loss_cls)).detach()
    
    def preprocess(self, targets, batch_size, scale_tensor):
        """preprocess gt bboxes and labels"""
        targets_list = jt.zeros((batch_size, 1, 5)).astype(targets.dtype)
        for i, item in enumerate(targets.cpu().numpy()):
            targets_list[i, :len(item)] = item
        return targets_list
    
    def bbox_decode(self, anchor_points, pred_dist):
        """decode pred dist to bbox"""
        if self.use_dfl:
            batch_size, n_anchors, _ = pred_dist.shape
            pred_dist = nn.softmax(pred_dist.view(batch_size, n_anchors, 4, self.reg_max + 1), dim=-1)
            pred_dist = pred_dist.matmul(self.proj.reshape([-1, 1])).squeeze(-1)
        
        return dist2bbox(pred_dist, anchor_points)


class VarifocalLoss(nn.Module):
    """Varifocal loss by Zhang et al. https://arxiv.org/abs/2008.13367."""
    
    def __init__(self):
        """Initialize the VarifocalLoss class."""
        super().__init__()
    
    def execute(self, pred_score, gt_score, label, alpha=0.75, gamma=2.0):
        """Computes varfocal loss."""
        weight = alpha * pred_score.sigmoid().pow(gamma) * (1 - label) + gt_score * label
        with jt.no_grad():
            # Jittor的二元交叉熵损失
            bce = -(label * jt.log(pred_score.sigmoid() + 1e-8) + 
                   (1 - label) * jt.log(1 - pred_score.sigmoid() + 1e-8))
        return (weight * bce).sum()


class BboxLoss(nn.Module):
    """Bbox loss computation."""
    
    def __init__(self, num_classes, reg_max, use_dfl=False, iou_type='giou'):
        """Initialize the BboxLoss module with regularization maximum, num_classes, and DFL settings."""
        super().__init__()
        self.num_classes = num_classes
        self.iou_loss = IOUloss(box_format='xyxy', iou_type=iou_type, eps=1e-10)
        self.reg_max = reg_max
        self.use_dfl = use_dfl
    
    def execute(self, pred_dist, pred_bboxes, anchor_points, target_bboxes, target_scores, target_scores_sum, fg_mask):
        """IoU loss."""
        weight = jt.masked_select(target_scores.sum(-1), fg_mask).unsqueeze(-1)
        iou = self.iou_loss(jt.masked_select(pred_bboxes, fg_mask.unsqueeze(-1)).view(-1, 4),
                           jt.masked_select(target_bboxes, fg_mask.unsqueeze(-1)).view(-1, 4))
        loss_iou = ((1.0 - iou) * weight).sum() / target_scores_sum
        
        # DFL loss
        if self.use_dfl:
            target_ltrb = bbox2dist(anchor_points, target_bboxes, self.reg_max)
            loss_dfl = self._df_loss(jt.masked_select(pred_dist, fg_mask.unsqueeze(-1)).view(-1, 4, self.reg_max + 1),
                                    jt.masked_select(target_ltrb, fg_mask.unsqueeze(-1)).view(-1, 4)) * weight
            loss_dfl = loss_dfl.sum() / target_scores_sum
        else:
            loss_dfl = jt.array(0.0)
        
        return loss_iou, loss_dfl
    
    @staticmethod
    def _df_loss(pred_dist, target):
        """Return sum of left and right DFL losses."""
        # Distribution Focal Loss (DFL) proposed in Generalized Focal Loss https://iccv.cc/2020/28
        tl = target.long()  # target left
        tr = tl + 1  # target right
        wl = tr - target  # weight left
        wr = 1 - wl  # weight right
        return (jt.nn.cross_entropy_loss(pred_dist, tl.view(-1), reduction="none").view(tl.shape) * wl +
                jt.nn.cross_entropy_loss(pred_dist, tr.view(-1), reduction="none").view(tl.shape) * wr).mean(-1, keepdim=True)

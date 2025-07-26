#!/usr/bin/env python3
# -*- coding:utf-8 -*-
"""
GOLD-YOLO Jittor版本 - FuseAB损失函数
严格对齐PyTorch版本，百分百还原所有功能
"""

import jittor as jt
import jittor.nn as nn
import numpy as np
from yolov6.assigners.atss_assigner import ATSSAssigner
from yolov6.assigners.tal_assigner import TaskAlignedAssigner
from yolov6.utils.figure_iou import IOUloss
from yolov6.utils.general import xywh2xyxy, bbox2dist


class VarifocalLoss(nn.Module):
    """Varifocal Loss for FuseAB - 百分百对齐PyTorch版本"""
    def __init__(self):
        super(VarifocalLoss, self).__init__()

    def execute(self, pred_score, gt_score, label, alpha=0.75, gamma=2.0):
        # 确保所有输入都是float32
        pred_score = pred_score.float32()
        gt_score = gt_score.float32()
        label = label.float32()
        
        weight = alpha * pred_score.pow(gamma) * (1 - label) + gt_score * label
        bce_loss = jt.nn.binary_cross_entropy_with_logits(pred_score.float(), gt_score.float())
        if len(bce_loss.shape) == 0:
            bce_loss = bce_loss.unsqueeze(0).expand_as(weight)
        elif len(bce_loss.shape) != len(weight.shape):
            bce_loss = bce_loss.expand_as(weight)
        
        focal_loss = weight * bce_loss
        return focal_loss


class BboxLoss(nn.Module):
    """Bbox Loss for FuseAB - 百分百对齐PyTorch版本"""
    def __init__(self, num_classes, reg_max, use_dfl=False, iou_type='giou'):
        super(BboxLoss, self).__init__()
        self.num_classes = num_classes
        self.iou_loss = IOUloss(box_format='xywh', iou_type=iou_type, eps=1e-10)
        self.reg_max = reg_max
        self.use_dfl = use_dfl
        if self.use_dfl:
            self.proj = jt.linspace(0, self.reg_max, self.reg_max + 1).float32().stop_grad()

    def execute(self, pred_dist, pred_bboxes, anchor_points, target_bboxes, target_scores, target_scores_sum, fg_mask):
        # 与主损失函数相同的实现
        pos_indices = jt.nonzero(fg_mask)
        
        if pos_indices.shape[0] > 0:
            pred_bboxes_pos = pred_bboxes[pos_indices[:, 0], pos_indices[:, 1]]
            target_bboxes_pos = target_bboxes[pos_indices[:, 0], pos_indices[:, 1]]
            bbox_weight = target_scores.sum(-1)[pos_indices[:, 0], pos_indices[:, 1]].unsqueeze(-1)
        else:
            pred_bboxes_pos = jt.zeros((0, 4), dtype='float32')
            target_bboxes_pos = jt.zeros((0, 4), dtype='float32')
            bbox_weight = jt.zeros((0, 1), dtype='float32')

        loss_iou = self.iou_loss(pred_bboxes_pos, target_bboxes_pos) * bbox_weight
        if target_scores_sum.item() == 0:
            loss_iou = loss_iou.sum()
        else:
            loss_iou = loss_iou.sum() / target_scores_sum

        # dfl loss
        if self.use_dfl and pos_indices.shape[0] > 0:
            pred_dist_pos = pred_dist[pos_indices[:, 0], pos_indices[:, 1]]
            pred_dist_pos = pred_dist_pos.reshape([-1, 4, self.reg_max + 1])
            
            target_ltrb = bbox2dist(anchor_points, target_bboxes, self.reg_max)
            target_ltrb_pos = target_ltrb[pos_indices[:, 0], pos_indices[:, 1]]
            
            loss_dfl = self._df_loss(pred_dist_pos, target_ltrb_pos) * bbox_weight
            
            if target_scores_sum.item() == 0:
                loss_dfl = loss_dfl.sum()
            else:
                loss_dfl = loss_dfl.sum() / target_scores_sum
        else:
            loss_dfl = pred_dist.sum() * 0.

        return loss_iou, loss_dfl

    def _df_loss(self, pred_dist, target):
        target_left = target.astype('int64')
        target_right = target_left + 1
        weight_left = target_right.astype('float32') - target
        weight_right = target - target_left.astype('float32')
        
        loss_left = jt.nn.cross_entropy_loss(pred_dist.view(-1, self.reg_max + 1), 
                                           target_left.view(-1), reduction='none').view(target_left.shape) * weight_left
        loss_right = jt.nn.cross_entropy_loss(pred_dist.view(-1, self.reg_max + 1), 
                                            target_right.view(-1), reduction='none').view(target_left.shape) * weight_right
        return (loss_left + loss_right).mean(-1, keepdims=True)


class ComputeLoss:
    '''FuseAB Loss computation func - 百分百还原PyTorch版本'''

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
        
        # FuseAB特定的assigner配置
        self.warmup_assigner = ATSSAssigner(topk=9, num_classes=self.num_classes)
        self.formal_assigner = TaskAlignedAssigner(topk=13, num_classes=self.num_classes, alpha=1.0, beta=6.0)

        self.use_dfl = use_dfl
        self.reg_max = reg_max
        self.proj = jt.linspace(0, self.reg_max, self.reg_max + 1).float32().stop_grad()
        self.iou_type = iou_type
        self.varifocal_loss = VarifocalLoss()
        self.bbox_loss = BboxLoss(self.num_classes, self.reg_max, self.use_dfl, self.iou_type)
        self.loss_weight = loss_weight

    def __call__(self, outputs, targets, epoch_num, step_num):
        """FuseAB损失计算 - 支持多分支输出"""
        # 处理FuseAB的多分支输出
        if isinstance(outputs, (list, tuple)) and len(outputs) > 3:
            # FuseAB有额外的分支输出
            feats, pred_scores, pred_distri = outputs[:3]
            aux_outputs = outputs[3:]  # 辅助输出
        else:
            feats, pred_scores, pred_distri = outputs
            aux_outputs = []

        # 主分支损失计算（与标准损失相同）
        main_loss, main_loss_items = self._compute_main_loss(
            feats, pred_scores, pred_distri, targets, epoch_num, step_num)

        # 辅助分支损失计算
        aux_loss = jt.zeros(1)
        if aux_outputs:
            for aux_output in aux_outputs:
                if isinstance(aux_output, (list, tuple)) and len(aux_output) >= 3:
                    aux_feats, aux_scores, aux_distri = aux_output[:3]
                    aux_loss_val, _ = self._compute_main_loss(
                        aux_feats, aux_scores, aux_distri, targets, epoch_num, step_num)
                    aux_loss += aux_loss_val * 0.5  # 辅助损失权重

        total_loss = main_loss + aux_loss
        return total_loss, main_loss_items

    def _compute_main_loss(self, feats, pred_scores, pred_distri, targets, epoch_num, step_num):
        """计算主损失 - 与标准ComputeLoss相同"""
        anchors, anchor_points, n_anchors_list, stride_tensor = \
               self.generate_anchors(feats, self.fpn_strides, self.grid_cell_size, self.grid_cell_offset)

        assert pred_scores.dtype == pred_distri.dtype
        gt_bboxes_scale = jt.full((1, 4), self.ori_img_size, dtype='float32')
        batch_size = pred_scores.shape[0]

        # targets
        targets = self.preprocess(targets, batch_size, gt_bboxes_scale)
        gt_labels = targets[:, :, :1]
        gt_bboxes = targets[:, :, 1:]
        mask_gt = (gt_labels.squeeze(-1) >= 0).astype('float32')

        # pboxes
        anchor_points_s = anchor_points / stride_tensor
        pred_bboxes = self.bbox_decode(anchor_points_s, pred_distri)

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
            print("assigner error, return high loss")
            return jt.ones(1, requires_grad=True) * 1000, jt.ones([4])

        # rescale bbox
        target_bboxes /= stride_tensor

        # cls loss
        target_labels = jt.where(fg_mask > 0, target_labels, jt.full_like(target_labels, self.num_classes))
        one_hot_label = jt.nn.one_hot(target_labels.astype('int64'), self.num_classes + 1)[..., :-1]
        loss_cls = self.varifocal_loss(pred_scores, target_scores, one_hot_label)

        target_scores_sum = target_scores.sum()
        if target_scores_sum.item() > 0:
            loss_cls /= target_scores_sum

        loss_cls = loss_cls.sum()

        # bbox loss
        loss_iou, loss_dfl = self.bbox_loss(pred_distri, pred_bboxes, anchor_points_s, target_bboxes, target_scores, target_scores_sum, fg_mask)

        loss = self.loss_weight['class'] * loss_cls + \
               self.loss_weight['iou'] * loss_iou + \
               self.loss_weight['dfl'] * loss_dfl

        return loss, jt.stack([loss_iou, loss_dfl, loss_cls]).detach()

    def generate_anchors(self, feats, fpn_strides, grid_cell_size=5.0, grid_cell_offset=0.5):
        """Generate anchors from features."""
        anchors = []
        anchor_points = []
        stride_tensor = []
        num_anchors_list = []
        
        for i, (feat, stride) in enumerate(zip(feats, fpn_strides)):
            _, _, h, w = feat.shape
            cell_half_size = grid_cell_size * grid_cell_offset
            shift_x = (jt.arange(w) + grid_cell_offset) * stride
            shift_y = (jt.arange(h) + grid_cell_offset) * stride
            shift_y, shift_x = jt.meshgrid(shift_y, shift_x)
            anchor = jt.stack([shift_x, shift_y], dim=-1).astype('float32')
            anchor_point = anchor.clone()
            anchor_point = anchor_point.reshape([-1, 2])
            anchor = anchor.unsqueeze(-2).expand([-1, -1, 1, -1])
            anchor = anchor.reshape([-1, 2])

            anchors.append(anchor)
            anchor_points.append(anchor_point)
            num_anchors_list.append(len(anchor_point))
            stride_tensor.append(jt.full([len(anchor_point), 1], stride, dtype='float32'))
        
        anchors = jt.concat(anchors)
        anchor_points = jt.concat(anchor_points).unsqueeze(0)
        stride_tensor = jt.concat(stride_tensor).unsqueeze(0)
        return anchors, anchor_points, num_anchors_list, stride_tensor

    def preprocess(self, targets, batch_size, scale_tensor):
        """Preprocess the targets."""
        targets_list = targets.numpy().tolist()
        max_len = max((len(l) for l in targets_list), default=1)
        targets_np = np.array(list(map(lambda l: l + [[-1, 0, 0, 0, 0]] * (max_len - len(l)), targets_list)), dtype=np.float32)[:, 1:, :]
        targets = jt.array(targets_np)
        scale_tensor = scale_tensor.float32()
        batch_target = targets[:, :, 1:5] * scale_tensor
        targets[..., 1:] = xywh2xyxy(batch_target)
        return targets

    def bbox_decode(self, anchor_points, pred_dist):
        """Decode predicted bbox."""
        if self.use_dfl:
            batch_size, n_anchors, _ = pred_dist.shape
            pred_dist = jt.nn.softmax(pred_dist.view(batch_size, n_anchors, 4, self.reg_max + 1), dim=-1)
            pred_dist = (pred_dist * self.proj.view(1, 1, 1, -1)).sum(-1)

        pred_dist = pred_dist.view(anchor_points.shape[0], anchor_points.shape[1], -1)
        pred_lt, pred_rb = pred_dist.split([2, 2], dim=-1)
        pred_x1y1 = anchor_points - pred_lt
        pred_x2y2 = anchor_points + pred_rb
        pred_bbox = jt.concat([pred_x1y1, pred_x2y2], dim=-1)
        return pred_bbox

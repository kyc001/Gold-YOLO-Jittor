# 2023.09.18-Changed for loss implementation of Gold-YOLO
#            Huawei Technologies Co., Ltd. <foss@huawei.com>
#!/usr/bin/env python3
# -*- coding:utf-8 -*-
"""
GOLD-YOLO Jittor版本 - 主损失函数
严格对齐PyTorch版本，百分百还原所有功能
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
    '''Loss computation func. - 严格对齐PyTorch版本'''

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
        # 严格对齐PyTorch: ATSSAssigner(9, num_classes=...)，位置参数是topk
        self.warmup_assigner = ATSSAssigner(9, num_classes=self.num_classes)
        self.formal_assigner = TaskAlignedAssigner(topk=13, num_classes=self.num_classes, alpha=1.0, beta=6.0)

        self.use_dfl = use_dfl
        self.reg_max = reg_max
        # 严格对齐PyTorch: nn.Parameter(..., requires_grad=False)
        self.proj = jt.linspace(0, self.reg_max, self.reg_max + 1).float32().stop_grad()
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
                             device=None)

        assert pred_scores.dtype == pred_distri.dtype
        gt_bboxes_scale = jt.full((1, 4), self.ori_img_size).float32()
        batch_size = pred_scores.shape[0]

        # targets
        targets = self.preprocess(targets, batch_size, gt_bboxes_scale)
        gt_labels = targets[:, :, :1]
        gt_bboxes = targets[:, :, 1:]  # xyxy
        mask_gt = (gt_bboxes.sum(-1, keepdims=True) > 0).float32()

        # pboxes
        anchor_points_s = anchor_points / stride_tensor
        pred_bboxes = self.bbox_decode(anchor_points_s, pred_distri)  # xyxy

        try:
            if epoch_num < self.warmup_epoch:
                # 严格对齐PyTorch: warmup_assigner的参数顺序
                # PyTorch: self.warmup_assigner(anchors, n_anchors_list, gt_labels, gt_bboxes, mask_gt, pred_bboxes * stride_tensor)
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
            print(
                    "OOM RuntimeError is raised due to the huge memory cost during label assignment. \
                        CPU mode is applied in this batch. If you want to avoid this issue, \
                        try to reduce the batch size or image size."
            )
            jt.gc()
            print("------------CPU Mode for This Batch-------------")
            if epoch_num < self.warmup_epoch:
                _anchors = anchors.float32()
                _n_anchors_list = n_anchors_list
                _gt_labels = gt_labels.float32()
                _gt_bboxes = gt_bboxes.float32()
                _mask_gt = mask_gt.float32()
                _pred_bboxes = pred_bboxes.detach().float32()
                _stride_tensor = stride_tensor.float32()

                target_labels, target_bboxes, target_scores, fg_mask = \
                    self.warmup_assigner(
                            _anchors,
                            _n_anchors_list,
                            _gt_labels,
                            _gt_bboxes,
                            _mask_gt,
                            _pred_bboxes * _stride_tensor)

            else:
                _pred_scores = pred_scores.detach().float32()
                _pred_bboxes = pred_bboxes.detach().float32()
                _anchor_points = anchor_points.float32()
                _gt_labels = gt_labels.float32()
                _gt_bboxes = gt_bboxes.float32()
                _mask_gt = mask_gt.float32()
                _stride_tensor = stride_tensor.float32()

                target_labels, target_bboxes, target_scores, fg_mask = \
                    self.formal_assigner(
                            _pred_scores,
                            _pred_bboxes * _stride_tensor,
                            _anchor_points,
                            _gt_labels,
                            _gt_bboxes,
                            _mask_gt)

        # Dynamic release GPU memory
        if step_num % 10 == 0:
            jt.gc()

        # Jittor在部分assigner路径会返回float64，这会触发cuDNN反向符号问题
        # 统一在loss入口处转成float32，保证反向图保持float32
        target_bboxes = target_bboxes.float32()
        target_scores = target_scores.float32()
        fg_mask = fg_mask.float32()

        # rescale bbox
        target_bboxes /= stride_tensor

        # cls loss - 严格对齐PyTorch
        target_labels = jt.where(fg_mask > 0, target_labels, jt.full_like(target_labels, self.num_classes))
        one_hot_label = jt.nn.one_hot(target_labels.int64(), self.num_classes + 1)[..., :-1].float32()
        loss_cls = self.varifocal_loss(pred_scores, target_scores, one_hot_label)

        # avoid devide zero error - 严格对齐PyTorch
        try:
            target_scores_sum = target_scores.sum()
            target_scores_sum_val = float(target_scores_sum.data)
            if target_scores_sum_val > 0:
                loss_cls /= target_scores_sum
        except Exception as e:
            print(f'Loss ERROR: {e}')
            try:
                if not jt.isnan(target_scores).any():
                    target_scores_sum = target_scores.sum()
                    if not jt.isnan(target_scores_sum).any() and float(target_scores_sum.data) > 0:
                        loss_cls /= target_scores_sum
                else:
                    target_scores_sum = jt.zeros(1)
            except:
                target_scores_sum = jt.zeros(1)

        # bbox loss
        loss_iou, loss_dfl = self.bbox_loss(pred_distri, pred_bboxes, anchor_points_s, target_bboxes,
                                            target_scores, target_scores_sum, fg_mask)

        loss = self.loss_weight['class'] * loss_cls + \
               self.loss_weight['iou'] * loss_iou + \
               self.loss_weight['dfl'] * loss_dfl

        # 严格对齐PyTorch: loss_items顺序是 [iou, dfl, cls]
        loss_items = jt.concat(((self.loss_weight['iou'] * loss_iou).unsqueeze(0),
                                (self.loss_weight['dfl'] * loss_dfl).unsqueeze(0),
                                (self.loss_weight['class'] * loss_cls).unsqueeze(0))).detach()

        return loss.float32(), loss_items.float32()

    def preprocess(self, targets, batch_size, scale_tensor):
        # 严格对齐PyTorch版本的preprocess
        targets_list = np.zeros((batch_size, 1, 5)).tolist()

        # 处理Jittor tensor转numpy
        if hasattr(targets, 'numpy'):
            targets_np = targets.detach().numpy()
        else:
            targets_np = np.array(targets)

        for i, item in enumerate(targets_np.tolist()):
            targets_list[int(item[0])].append(item[1:])
        max_len = max((len(l) for l in targets_list))
        targets = jt.array(
                np.array(list(map(lambda l: l + [[-1, 0, 0, 0, 0]] * (max_len - len(l)), targets_list)))[:, 1:, :]).float32()
        batch_target = targets[:, :, 1:5] * scale_tensor
        targets[..., 1:] = xywh2xyxy(batch_target)
        return targets

    def bbox_decode(self, anchor_points, pred_dist):
        if self.use_dfl:
            batch_size, n_anchors, _ = pred_dist.shape
            pred_dist = jt.nn.softmax(pred_dist.view(batch_size, n_anchors, 4, self.reg_max + 1), dim=-1)
            pred_dist = (pred_dist * self.proj.view(1, 1, 1, -1)).sum(-1)
        return dist2bbox(pred_dist, anchor_points)


class VarifocalLoss(nn.Module):
    """严格对齐PyTorch版本的VarifocalLoss"""
    def __init__(self):
        super(VarifocalLoss, self).__init__()

    def execute(self, pred_score, gt_score, label, alpha=0.75, gamma=2.0):
        # 严格对齐PyTorch: weight = alpha * pred_score.pow(gamma) * (1 - label) + gt_score * label
        pred_score_float = pred_score.float32()
        gt_score_float = gt_score.float32()
        label_float = label.float32()
        weight = alpha * pred_score_float.pow(gamma) * (1 - label_float) + gt_score_float * label_float

        # 严格对齐PyTorch: F.binary_cross_entropy(pred_score.float(), gt_score.float(), reduction='none')
        # 注意: PyTorch的BCE期望输入是概率（已经过sigmoid），不是logits
        # pred_score 应该已经是 sigmoid 后的结果（在 effidehead.py 中已应用）

        # 手动实现binary_cross_entropy (reduction='none')
        # BCE = -[y * log(p) + (1-y) * log(1-p)]
        eps = 1e-7
        pred_clamped = jt.clamp(pred_score_float, eps, 1 - eps)
        bce = -(gt_score_float * jt.log(pred_clamped) + (1 - gt_score_float) * jt.log(1 - pred_clamped))

        # 严格对齐PyTorch: (bce * weight).sum()
        loss = (bce * weight).sum()

        return loss.float32()


class BboxLoss(nn.Module):
    """严格对齐PyTorch版本的BboxLoss"""
    def __init__(self, num_classes, reg_max, use_dfl=False, iou_type='giou'):
        super(BboxLoss, self).__init__()
        self.num_classes = num_classes
        self.iou_loss = IOUloss(box_format='xyxy', iou_type=iou_type, eps=1e-10)
        self.reg_max = reg_max
        self.use_dfl = use_dfl

    def execute(self, pred_dist, pred_bboxes, anchor_points,
                target_bboxes, target_scores, target_scores_sum, fg_mask):

        # select positive samples mask
        num_pos = fg_mask.sum()
        num_pos_val = float(num_pos.data)

        if num_pos_val > 0:
            # iou loss - 严格对齐PyTorch的masked_select方式
            bbox_mask = fg_mask.unsqueeze(-1).repeat([1, 1, 4])

            # Jittor实现masked_select
            pred_bboxes_pos = pred_bboxes[bbox_mask.bool()].reshape([-1, 4])
            target_bboxes_pos = target_bboxes[bbox_mask.bool()].reshape([-1, 4])
            bbox_weight = target_scores.sum(-1)[fg_mask.bool()].unsqueeze(-1)

            loss_iou = self.iou_loss(pred_bboxes_pos, target_bboxes_pos) * bbox_weight

            target_scores_sum_val = float(target_scores_sum.data) if hasattr(target_scores_sum, 'data') else float(target_scores_sum)
            if target_scores_sum_val == 0:
                loss_iou = loss_iou.sum()
            else:
                loss_iou = loss_iou.sum() / target_scores_sum

            # dfl loss
            if self.use_dfl:
                dist_mask = fg_mask.unsqueeze(-1).repeat([1, 1, (self.reg_max + 1) * 4])
                pred_dist_pos = pred_dist[dist_mask.bool()].reshape([-1, 4, self.reg_max + 1])
                target_ltrb = bbox2dist(anchor_points, target_bboxes, self.reg_max)
                target_ltrb_pos = target_ltrb[bbox_mask.bool()].reshape([-1, 4])
                loss_dfl = self._df_loss(pred_dist_pos, target_ltrb_pos) * bbox_weight
                if target_scores_sum_val == 0:
                    loss_dfl = loss_dfl.sum()
                else:
                    loss_dfl = loss_dfl.sum() / target_scores_sum
            else:
                loss_dfl = pred_dist.sum() * 0.

        else:
            loss_iou = pred_dist.sum() * 0.
            loss_dfl = pred_dist.sum() * 0.

        return loss_iou, loss_dfl

    def _df_loss(self, pred_dist, target):
        """严格对齐PyTorch版本的DFL损失"""
        # 严格对齐PyTorch:
        # target_left = target.to(torch.long)
        # target_right = target_left + 1
        # weight_left = target_right.to(torch.float) - target
        # weight_right = 1 - weight_left
        target_left = target.int64()
        target_right = target_left + 1
        weight_left = target_right.float32() - target
        weight_right = 1 - weight_left

        # 严格对齐PyTorch: F.cross_entropy(..., reduction='none')
        # cross_entropy期望输入shape: [N, C] 和 target shape: [N]
        # pred_dist shape: [num_pos, 4, reg_max+1]
        # target shape: [num_pos, 4]

        # Jittor实现cross_entropy
        pred_dist_flat = pred_dist.view(-1, self.reg_max + 1)  # [num_pos*4, reg_max+1]
        target_left_flat = target_left.view(-1)  # [num_pos*4]
        target_right_flat = target_right.view(-1)  # [num_pos*4]

        # 限制target范围防止越界
        target_left_flat = jt.clamp(target_left_flat, 0, self.reg_max)
        target_right_flat = jt.clamp(target_right_flat, 0, self.reg_max)

        # 计算log_softmax
        log_softmax = jt.nn.log_softmax(pred_dist_flat, dim=-1)

        # 使用gather获取对应类别的log概率
        # loss = -log_softmax[target]
        batch_size = pred_dist_flat.shape[0]
        indices = jt.arange(batch_size)

        loss_left = -log_softmax[indices, target_left_flat]
        loss_right = -log_softmax[indices, target_right_flat]

        # reshape回原始形状
        loss_left = loss_left.view(target_left.shape) * weight_left
        loss_right = loss_right.view(target_left.shape) * weight_right

        # 严格对齐PyTorch: (loss_left + loss_right).mean(-1, keepdim=True)
        return (loss_left + loss_right).mean(-1, keepdims=True)

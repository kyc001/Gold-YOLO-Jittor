"""
GOLD-YOLO Jittor版本 - 分配器工具函数
严格对齐PyTorch版本
"""

import jittor as jt
import jittor.nn as nn


def dist_calculator(gt_bboxes, anchor_bboxes):
    """compute center distance between all bbox and gt
    严格对齐PyTorch版本

    Args:
        gt_bboxes (Tensor): shape(bs*n_max_boxes, 4)
        anchor_bboxes (Tensor): shape(num_total_anchors, 4)
    Return:
        distances (Tensor): shape(bs*n_max_boxes, num_total_anchors)
        ac_points (Tensor): shape(num_total_anchors, 2)
    """
    gt_cx = (gt_bboxes[:, 0] + gt_bboxes[:, 2]) / 2.0
    gt_cy = (gt_bboxes[:, 1] + gt_bboxes[:, 3]) / 2.0
    gt_points = jt.stack([gt_cx, gt_cy], dim=1)
    ac_cx = (anchor_bboxes[:, 0] + anchor_bboxes[:, 2]) / 2.0
    ac_cy = (anchor_bboxes[:, 1] + anchor_bboxes[:, 3]) / 2.0
    ac_points = jt.stack([ac_cx, ac_cy], dim=1)

    # 严格对齐PyTorch: (gt_points[:, None, :] - ac_points[None, :, :]).pow(2).sum(-1).sqrt()
    distances = (gt_points[:, None, :] - ac_points[None, :, :]).pow(2).sum(-1).sqrt()

    return distances, ac_points


def select_candidates_in_gts(xy_centers, gt_bboxes, eps=1e-9):
    """select the positive anchors's center in gt
    严格对齐PyTorch版本

    Args:
        xy_centers (Tensor): shape(num_total_anchors, 2)
        gt_bboxes (Tensor): shape(bs, n_max_boxes, 4)
    Return:
        (Tensor): shape(bs, n_max_boxes, num_total_anchors)
    """
    n_anchors = xy_centers.size(0)
    bs, n_max_boxes, _ = gt_bboxes.size()
    _gt_bboxes = gt_bboxes.reshape([-1, 4])
    xy_centers = xy_centers.unsqueeze(0).repeat(bs * n_max_boxes, 1, 1)
    gt_bboxes_lt = _gt_bboxes[:, 0:2].unsqueeze(1).repeat(1, n_anchors, 1)
    gt_bboxes_rb = _gt_bboxes[:, 2:4].unsqueeze(1).repeat(1, n_anchors, 1)
    b_lt = xy_centers - gt_bboxes_lt
    b_rb = gt_bboxes_rb - xy_centers
    bbox_deltas = jt.concat([b_lt, b_rb], dim=-1)
    bbox_deltas = bbox_deltas.reshape([bs, n_max_boxes, n_anchors, -1])
    return (bbox_deltas.min(dim=-1)[0] > eps).astype(gt_bboxes.dtype)


def select_highest_overlaps(mask_pos, overlaps, n_max_boxes):
    """if an anchor box is assigned to multiple gts,
        the one with the highest iou will be selected.
    严格对齐PyTorch版本

    Args:
        mask_pos (Tensor): shape(bs, n_max_boxes, num_total_anchors)
        overlaps (Tensor): shape(bs, n_max_boxes, num_total_anchors)
    Return:
        target_gt_idx (Tensor): shape(bs, num_total_anchors)
        fg_mask (Tensor): shape(bs, num_total_anchors)
        mask_pos (Tensor): shape(bs, n_max_boxes, num_total_anchors)
    """
    # 严格对齐PyTorch: fg_mask = mask_pos.sum(axis=-2)
    # Jittor不支持负数dim，手动转换: dim=-2 对于3D张量等于 dim=1
    fg_mask = mask_pos.sum(dim=1)
    if fg_mask.max() > 1:
        mask_multi_gts = (fg_mask.unsqueeze(1) > 1).repeat([1, n_max_boxes, 1])
        # 严格对齐PyTorch: max_overlaps_idx = overlaps.argmax(axis=1)
        # Jittor的argmax返回(indices, values)元组，需要取[0]获取索引
        max_overlaps_idx = overlaps.argmax(dim=1)[0]
        is_max_overlaps = nn.one_hot(max_overlaps_idx, n_max_boxes)
        is_max_overlaps = is_max_overlaps.permute(0, 2, 1).astype(overlaps.dtype)
        mask_pos = jt.where(mask_multi_gts, is_max_overlaps, mask_pos)
        fg_mask = mask_pos.sum(dim=1)
    # 严格对齐PyTorch: target_gt_idx = mask_pos.argmax(axis=-2)
    # Jittor不支持负数dim，dim=-2 对于3D张量等于 dim=1
    # Jittor的argmax返回(indices, values)元组，需要取[0]获取索引
    target_gt_idx = mask_pos.argmax(dim=1)[0]
    return target_gt_idx, fg_mask, mask_pos


def iou_calculator(box1, box2, eps=1e-9):
    """Calculate iou for batch
    严格对齐PyTorch版本

    Args:
        box1 (Tensor): shape(bs, n_max_boxes, 4)
        box2 (Tensor): shape(bs, num_total_anchors, 4)
    Return:
        (Tensor): shape(bs, n_max_boxes, num_total_anchors)
    """
    box1 = box1.unsqueeze(2)  # [N, M1, 4] -> [N, M1, 1, 4]
    box2 = box2.unsqueeze(1)  # [N, M2, 4] -> [N, 1, M2, 4]
    px1y1, px2y2 = box1[:, :, :, 0:2], box1[:, :, :, 2:4]
    gx1y1, gx2y2 = box2[:, :, :, 0:2], box2[:, :, :, 2:4]
    x1y1 = jt.maximum(px1y1, gx1y1)
    x2y2 = jt.minimum(px2y2, gx2y2)
    overlap = (x2y2 - x1y1).clamp(0).prod(-1)
    area1 = (px2y2 - px1y1).clamp(0).prod(-1)
    area2 = (gx2y2 - gx1y1).clamp(0).prod(-1)
    union = area1 + area2 - overlap + eps

    return overlap / union

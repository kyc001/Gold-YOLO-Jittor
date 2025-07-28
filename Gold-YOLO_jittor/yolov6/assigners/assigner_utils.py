"""
GOLD-YOLO Jittor版本 - 分配器工具函数
从PyTorch版本迁移到Jittor框架
"""

import jittor as jt
from jittor import nn
import jittor.nn as nn


def dist_calculator(gt_bboxes, anchor_bboxes):
    """compute center distance between all bbox and gt

    Args:
        gt_bboxes (Tensor): shape(bs*n_max_boxes, 4)
        anchor_bboxes (Tensor): shape(num_total_anchors, 4)
    Return:
        distances (Tensor): shape(bs*n_max_boxes, num_total_anchors)
        ac_points (Tensor): shape(num_total_anchors, 2)
    """
    # 确保anchor_bboxes是2D的[M, 4]格式
    if len(anchor_bboxes.shape) != 2 or anchor_bboxes.shape[-1] != 4:
        # 尝试reshape
        if anchor_bboxes.numel() % 4 == 0:
            anchor_bboxes = anchor_bboxes.view(-1, 4)
        else:
            raise ValueError(f"anchor_bboxes无法reshape为[M, 4]: {anchor_bboxes.shape}")

    gt_cx = (gt_bboxes[:, 0] + gt_bboxes[:, 2]) / 2.0
    gt_cy = (gt_bboxes[:, 1] + gt_bboxes[:, 3]) / 2.0
    gt_points = jt.stack([gt_cx, gt_cy], dim=1)
    ac_cx = (anchor_bboxes[:, 0] + anchor_bboxes[:, 2]) / 2.0
    ac_cy = (anchor_bboxes[:, 1] + anchor_bboxes[:, 3]) / 2.0
    ac_points = jt.stack([ac_cx, ac_cy], dim=1)

    # 修复形状不匹配问题
    # gt_points: [N, 2], ac_points: [M, 2]
    # 需要计算每个GT点到每个anchor点的距离

    N = gt_points.shape[0]
    M = ac_points.shape[0]

    # 扩展维度进行广播
    gt_points_exp = gt_points.unsqueeze(1)  # [N, 1, 2]
    ac_points_exp = ac_points.unsqueeze(0)  # [1, M, 2]

    # 计算距离
    distances = ((gt_points_exp - ac_points_exp) ** 2).sum(-1).sqrt()  # [N, M]

    return distances, ac_points


def select_candidates_in_gts(xy_centers, gt_bboxes, eps=1e-9):
    """select the positive anchors's center in gt

    Args:
        xy_centers (Tensor): shape(bs*n_max_boxes, num_total_anchors, 4) - anchor中心点坐标(像素坐标)
        gt_bboxes (Tensor): shape(bs, n_max_boxes, 4) - GT框坐标(可能是归一化坐标)
    Return:
        (Tensor): shape(bs, n_max_boxes, num_total_anchors)
    """
    # 调试：检查输入格式
    print(f"🔍 [GT内检查] xy_centers形状: {xy_centers.shape}, 数值范围: [{float(xy_centers.min().data):.1f}, {float(xy_centers.max().data):.1f}]")
    print(f"🔍 [GT内检查] gt_bboxes形状: {gt_bboxes.shape}, 数值范围: [{float(gt_bboxes.min().data):.1f}, {float(gt_bboxes.max().data):.1f}]")

    # 检查坐标系统并修复不匹配问题
    if gt_bboxes.shape[1] > 0:
        first_gt = gt_bboxes[0, 0].numpy()
        width = first_gt[2] - first_gt[0]
        height = first_gt[3] - first_gt[1]
        print(f"🔍 [GT内检查] 第一个GT框(原始): {first_gt}")
        print(f"🔍 [GT内检查] 第一个GT框尺寸(原始): 宽={width:.6f}, 高={height:.6f}")

        # 检测坐标系统：如果GT框坐标都在[0,1]范围内，说明是归一化坐标
        max_coord = float(gt_bboxes.max().data)
        if max_coord <= 1.0:
            print(f"🔧 [坐标修复] 检测到归一化坐标，转换为像素坐标")
            # 将GT框从归一化坐标转换为像素坐标
            gt_bboxes = gt_bboxes * 640.0
            first_gt_fixed = gt_bboxes[0, 0].numpy()
            width_fixed = first_gt_fixed[2] - first_gt_fixed[0]
            height_fixed = first_gt_fixed[3] - first_gt_fixed[1]
            print(f"🔧 [坐标修复] 第一个GT框(修复后): {first_gt_fixed}")
            print(f"🔧 [坐标修复] 第一个GT框尺寸(修复后): 宽={width_fixed:.1f}, 高={height_fixed:.1f}")

        print(f"🔍 [GT内检查] 前3个anchor点: {xy_centers[:3, 0].numpy()}")

        # 重新统计目标尺寸(使用像素坐标)
        all_widths = gt_bboxes[0, :, 2] - gt_bboxes[0, :, 0]
        all_heights = gt_bboxes[0, :, 3] - gt_bboxes[0, :, 1]
        small_targets = ((all_widths < 1.0) | (all_heights < 1.0)).sum()
        print(f"🔍 [GT内检查] 极小目标数量(宽或高<1像素): {int(small_targets.data)}/{gt_bboxes.shape[1]}")

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

    # 调试：检查bbox_deltas
    min_deltas = bbox_deltas.min(dim=-1)[0]
    print(f"🔍 [GT内检查] bbox_deltas最小值范围: [{float(min_deltas.min().data):.6f}, {float(min_deltas.max().data):.6f}]")
    print(f"🔍 [GT内检查] eps阈值: {eps}")

    result = (min_deltas > eps).astype(gt_bboxes.dtype)
    print(f"🔍 [GT内检查] 结果: 在GT内的anchor数量: {int(result.sum().data)}")

    return result


def select_highest_overlaps(mask_pos, overlaps, n_max_boxes):
    """if an anchor box is assigned to multiple gts,
        the one with the highest iou will be selected.

    Args:
        mask_pos (Tensor): shape(bs, n_max_boxes, num_total_anchors)
        overlaps (Tensor): shape(bs, n_max_boxes, num_total_anchors)
    Return:
        target_gt_idx (Tensor): shape(bs, num_total_anchors)
        fg_mask (Tensor): shape(bs, num_total_anchors)
        mask_pos (Tensor): shape(bs, n_max_boxes, num_total_anchors)
    """
    fg_mask = mask_pos.sum(dim=-2)
    if fg_mask.max() > 1:
        mask_multi_gts = (fg_mask.unsqueeze(1) > 1).repeat([1, n_max_boxes, 1])
        max_overlaps_idx = jt.argmax(overlaps, dim=1)[0]  # 修复Jittor argmax API
        is_max_overlaps = nn.one_hot(max_overlaps_idx, n_max_boxes)
        is_max_overlaps = is_max_overlaps.permute(0, 2, 1).astype(overlaps.dtype)
        mask_pos = jt.where(mask_multi_gts, is_max_overlaps, mask_pos)
        fg_mask = mask_pos.sum(dim=-2)
    # 修复Jittor argmax API - 使用正确的维度索引
    target_gt_idx = jt.argmax(mask_pos, dim=1)[0]  # Jittor返回(indices, values)
    return target_gt_idx, fg_mask , mask_pos


def iou_calculator(box1, box2, eps=1e-9):
    """Calculate iou for batch

    Args:
        box1 (Tensor): shape(bs, n_max_boxes, 1, 4) 或 (bs, n_max_boxes, 4)
        box2 (Tensor): shape(bs, 1, num_total_anchors, 4) 或 (bs, num_total_anchors, 4)
    Return:
        (Tensor): shape(bs, n_max_boxes, num_total_anchors)
    """
    # 确保正确的维度 - 如果输入已经有正确维度就不要再unsqueeze
    if len(box1.shape) == 3:  # [bs, n_max_boxes, 4]
        box1 = box1.unsqueeze(2)  # -> [bs, n_max_boxes, 1, 4]
    if len(box2.shape) == 3:  # [bs, num_total_anchors, 4]
        box2 = box2.unsqueeze(1)  # -> [bs, 1, num_total_anchors, 4]
    # 修复Jittor 4维切片问题 - 使用split替代切片
    px1y1, px2y2 = jt.split(box1, [2, 2], dim=-1)
    gx1y1, gx2y2 = jt.split(box2, [2, 2], dim=-1)
    x1y1 = jt.maximum(px1y1, gx1y1)
    x2y2 = jt.minimum(px2y2, gx2y2)
    overlap = (x2y2 - x1y1).clamp(0).prod(-1)
    area1 = (px2y2 - px1y1).clamp(0).prod(-1)
    area2 = (gx2y2 - gx1y1).clamp(0).prod(-1)
    union = area1 + area2 - overlap + eps

    return overlap / union


def bbox_overlaps(bboxes1, bboxes2, mode='iou', is_aligned=False, eps=1e-6):
    """Calculate overlap between two set of bboxes.
    
    Args:
        bboxes1 (Tensor): shape (B, m, 4) in <x1, y1, x2, y2> format or empty.
        bboxes2 (Tensor): shape (B, n, 4) in <x1, y1, x2, y2> format or empty.
        mode (str): "iou" (intersection over union), "iof" (intersection over foreground)
        is_aligned (bool): If True, then m and n must be equal.
        eps (float): A value added to the denominator for numerical stability.
    
    Returns:
        Tensor: shape (m, n) if ``is_aligned`` is False else shape (m,)
    """
    assert mode in ['iou', 'iof', 'giou'], f'Unsupported mode {mode}'
    # Either the boxes are empty or the length of boxes' last dimension is 4
    assert (bboxes1.size(-1) == 4 or bboxes1.size(0) == 0)
    assert (bboxes2.size(-1) == 4 or bboxes2.size(0) == 0)

    # Batch dim must be the same
    # Batch dim: (B1, B2, ... Bn)
    assert bboxes1.shape[:-2] == bboxes2.shape[:-2]
    batch_shape = bboxes1.shape[:-2]

    rows = bboxes1.size(-2)
    cols = bboxes2.size(-2)
    if is_aligned:
        assert rows == cols

    if rows * cols == 0:
        if is_aligned:
            return bboxes1.new(batch_shape + (rows, ))
        else:
            return bboxes1.new(batch_shape + (rows, cols))

    area1 = (bboxes1[..., 2] - bboxes1[..., 0]) * (
        bboxes1[..., 3] - bboxes1[..., 1])
    area2 = (bboxes2[..., 2] - bboxes2[..., 0]) * (
        bboxes2[..., 3] - bboxes2[..., 1])

    if is_aligned:
        lt = jt.maximum(bboxes1[..., :2], bboxes2[..., :2])  # [B, rows, 2]
        rb = jt.minimum(bboxes1[..., 2:], bboxes2[..., 2:])  # [B, rows, 2]

        wh = jt.maximum(rb - lt, 0)  # [B, rows, 2]
        overlap = wh[..., 0] * wh[..., 1]

        if mode in ['iou', 'giou']:
            union = area1 + area2 - overlap
        else:
            union = area1
        if mode == 'giou':
            enclosed_lt = jt.minimum(bboxes1[..., :2], bboxes2[..., :2])
            enclosed_rb = jt.maximum(bboxes1[..., 2:], bboxes2[..., 2:])
    else:
        lt = jt.maximum(bboxes1[..., :, None, :2],
                       bboxes2[..., None, :, :2])  # [B, rows, cols, 2]
        rb = jt.minimum(bboxes1[..., :, None, 2:],
                       bboxes2[..., None, :, 2:])  # [B, rows, cols, 2]

        wh = jt.maximum(rb - lt, 0)  # [B, rows, cols, 2]
        overlap = wh[..., 0] * wh[..., 1]

        if mode in ['iou', 'giou']:
            union = area1[..., :, None] + area2[..., None, :] - overlap
        else:
            union = area1[..., :, None]
        if mode == 'giou':
            enclosed_lt = jt.minimum(bboxes1[..., :, None, :2],
                                    bboxes2[..., None, :, :2])
            enclosed_rb = jt.maximum(bboxes1[..., :, None, 2:],
                                    bboxes2[..., None, :, 2:])

    eps = union.new_tensor([eps])
    union = jt.maximum(union, eps)
    ious = overlap / union
    if mode in ['iou', 'iof']:
        return ious
    # calculate gious
    enclose_wh = jt.maximum(enclosed_rb - enclosed_lt, 0)
    enclose_area = enclose_wh[..., 0] * enclose_wh[..., 1]
    enclose_area = jt.maximum(enclose_area, eps)
    gious = ious - (enclose_area - union) / enclose_area
    return gious

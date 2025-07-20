#!/usr/bin/env python3
# -*- coding:utf-8 -*-
"""
General utilities for YOLOv6 Jittor implementation
"""

import jittor as jt
import numpy as np


def dist2bbox(distance, anchor_points, box_format='xyxy'):
    """Transform distance(ltrb) to box(xywh or xyxy)."""
    lt, rb = distance.chunk(2, -1)
    x1y1 = anchor_points - lt
    x2y2 = anchor_points + rb
    if box_format == 'xyxy':
        bbox = jt.concat([x1y1, x2y2], -1)
    elif box_format == 'xywh':
        c_xy = (x1y1 + x2y2) / 2
        wh = x2y2 - x1y1
        bbox = jt.concat([c_xy, wh], -1)
    return bbox


def generate_anchors(feats, fpn_strides, grid_cell_size=5.0, grid_cell_offset=0.5, is_eval=False, mode='af'):
    """Generate anchors from features - 完全对齐PyTorch版本"""
    anchors = []
    anchor_points = []
    stride_tensor = []
    num_anchors_list = []
    assert feats is not None

    if is_eval:
        for i, stride in enumerate(fpn_strides):
            _, _, h, w = feats[i].shape
            shift_x = jt.arange(end=w) + grid_cell_offset
            shift_y = jt.arange(end=h) + grid_cell_offset
            shift_y, shift_x = jt.meshgrid(shift_y, shift_x)
            anchor_point = jt.stack([shift_x, shift_y], dim=-1).float()

            if mode == 'af':  # anchor-free
                anchor_points.append(anchor_point.reshape([-1, 2]))
                stride_tensor.append(jt.full((h * w, 1), stride, dtype=jt.float32))
            elif mode == 'ab':  # anchor-based
                # 修复Jittor repeat调用 - 使用正确的语法
                reshaped_points = anchor_point.reshape([-1, 2])
                anchor_points.append(reshaped_points.repeat(3, 1))
                stride_full = jt.full((h * w, 1), stride, dtype=jt.float32)
                stride_tensor.append(stride_full.repeat(3, 1))

        anchor_points = jt.concat(anchor_points)
        stride_tensor = jt.concat(stride_tensor)
        return anchor_points, stride_tensor
    else:
        for i, stride in enumerate(fpn_strides):
            _, _, h, w = feats[i].shape
            cell_half_size = grid_cell_size * stride * 0.5
            shift_x = (jt.arange(end=w) + grid_cell_offset) * stride
            shift_y = (jt.arange(end=h) + grid_cell_offset) * stride
            shift_y, shift_x = jt.meshgrid(shift_y, shift_x)

            anchor = jt.stack([
                shift_x - cell_half_size, shift_y - cell_half_size,
                shift_x + cell_half_size, shift_y + cell_half_size
            ], dim=-1).clone().to(feats[0].dtype)

            anchor_point = jt.stack([shift_x, shift_y], dim=-1).clone().to(feats[0].dtype)

            if mode == 'af':  # anchor-free
                anchors.append(anchor.reshape([-1, 4]))
                anchor_points.append(anchor_point.reshape([-1, 2]))
                stride_tensor.append(jt.full((h * w, 1), stride, dtype=feats[0].dtype))
                num_anchors_list.append(anchors[-1].shape[0])
            elif mode == 'ab':  # anchor-based
                # 修复Jittor repeat调用 - 使用正确的语法
                reshaped_anchors = anchor.reshape([-1, 4])
                anchors.append(reshaped_anchors.repeat(3, 1))
                reshaped_points = anchor_point.reshape([-1, 2])
                anchor_points.append(reshaped_points.repeat(3, 1))
                stride_full = jt.full((h * w, 1), stride, dtype=feats[0].dtype)
                stride_tensor.append(stride_full.repeat(3, 1))
                num_anchors_list.append(anchors[-1].shape[0])

        anchors = jt.concat(anchors)
        anchor_points = jt.concat(anchor_points)
        stride_tensor = jt.concat(stride_tensor)
        return anchors, anchor_points, stride_tensor, num_anchors_list


def make_anchors(feats, strides, grid_cell_offset=0.5):
    """简化版本的anchor生成 - 向后兼容"""
    anchor_points, stride_tensor = generate_anchors(feats, strides, grid_cell_offset=grid_cell_offset, is_eval=True, mode='af')
    return anchor_points, stride_tensor


def bbox2dist(anchor_points, bbox, reg_max):
    """Transform bbox(xyxy) to dist(ltrb)."""
    x1y1, x2y2 = bbox.chunk(2, -1)
    return jt.concat((anchor_points - x1y1, x2y2 - anchor_points), -1).clamp(0, reg_max - 0.01)


def xywh2xyxy(x):
    """Convert nx4 boxes from [x, y, w, h] to [x1, y1, x2, y2] where xy1=top-left, xy2=bottom-right."""
    y = x.clone() if isinstance(x, jt.Var) else jt.array(x)
    y[..., 0] = x[..., 0] - x[..., 2] / 2  # top left x
    y[..., 1] = x[..., 1] - x[..., 3] / 2  # top left y
    y[..., 2] = x[..., 0] + x[..., 2] / 2  # bottom right x
    y[..., 3] = x[..., 1] + x[..., 3] / 2  # bottom right y
    return y


def box_iou(box1, box2, eps=1e-7):
    """计算IoU - 完全对齐PyTorch版本"""
    # box1: [N, 4]
    # box2: [M, 4]

    # 扩展维度
    box1 = box1.unsqueeze(1)  # [N, 1, 4]
    box2 = box2.unsqueeze(0)  # [1, M, 4]

    # 计算交集
    lt = jt.maximum(box1[..., :2], box2[..., :2])  # [N, M, 2]
    rb = jt.minimum(box1[..., 2:], box2[..., 2:])  # [N, M, 2]

    wh = (rb - lt).clamp(min_v=0)  # [N, M, 2]
    inter = wh[..., 0] * wh[..., 1]  # [N, M]

    # 计算面积
    area1 = (box1[..., 2] - box1[..., 0]) * (box1[..., 3] - box1[..., 1])  # [N, 1]
    area2 = (box2[..., 2] - box2[..., 0]) * (box2[..., 3] - box2[..., 1])  # [1, M]

    union = area1 + area2 - inter + eps  # [N, M]

    return inter / union  # [N, M]


def xyxy2xywh(x):
    """Convert nx4 boxes from [x1, y1, x2, y2] to [x, y, w, h] where xy1=top-left, xy2=bottom-right."""
    y = x.clone() if isinstance(x, jt.Var) else jt.array(x)
    y[:, 0] = (x[:, 0] + x[:, 2]) / 2  # x center
    y[:, 1] = (x[:, 1] + x[:, 3]) / 2  # y center
    y[:, 2] = x[:, 2] - x[:, 0]  # width
    y[:, 3] = x[:, 3] - x[:, 1]  # height
    return y


def scale_coords(img1_shape, coords, img0_shape, ratio_pad=None):
    """Rescale coords (xyxy) from img1_shape to img0_shape."""
    if ratio_pad is None:  # calculate from img0_shape
        gain = min(img1_shape[0] / img0_shape[0], img1_shape[1] / img0_shape[1])  # gain  = old / new
        pad = (img1_shape[1] - img0_shape[1] * gain) / 2, (img1_shape[0] - img0_shape[0] * gain) / 2  # wh padding
    else:
        gain = ratio_pad[0][0]
        pad = ratio_pad[1]

    coords[:, [0, 2]] -= pad[0]  # x padding
    coords[:, [1, 3]] -= pad[1]  # y padding
    coords[:, :4] /= gain
    clip_coords(coords, img0_shape)
    return coords


def clip_coords(boxes, shape):
    """Clip bounding xyxy bounding boxes to image shape (height, width)."""
    boxes[:, [0, 2]] = boxes[:, [0, 2]].clamp(min_v=0, max_v=shape[1])  # x1, x2
    boxes[:, [1, 3]] = boxes[:, [1, 3]].clamp(min_v=0, max_v=shape[0])  # y1, y2


def non_max_suppression_jittor(prediction, conf_thres=0.25, iou_thres=0.45, classes=None, agnostic=False, multi_label=False, max_det=300):
    """Runs Non-Maximum Suppression (NMS) on inference results - 简化Jittor实现"""

    output = []

    for img_idx, x in enumerate(prediction):  # image index, image inference
        # 简化的置信度过滤
        obj_conf = x[:, 4]  # 目标置信度
        cls_conf = jt.max(x[:, 5:], dim=1)[0]  # 最大类别置信度
        final_conf = obj_conf * cls_conf  # 最终置信度

        # 过滤低置信度
        valid_mask = final_conf > conf_thres
        if not valid_mask.any():
            output.append(jt.zeros((0, 6)))
            continue

        # 获取有效检测
        valid_x = x[valid_mask]
        valid_conf = final_conf[valid_mask]

        # 获取类别
        cls_idx = jt.argmax(valid_x[:, 5:], dim=1)[0]

        # 转换边界框格式
        boxes = xywh2xyxy(valid_x[:, :4])

        # 构建输出：[x1, y1, x2, y2, conf, cls]
        detections = jt.zeros((valid_x.shape[0], 6))
        detections[:, :4] = boxes
        detections[:, 4] = valid_conf
        detections[:, 5] = cls_idx.float()

        # 按置信度排序，保留前max_det个
        if detections.shape[0] > max_det:
            sorted_indices = valid_conf.argsort(descending=True)[0][:max_det]
            detections = detections[sorted_indices]

        output.append(detections)

    return output


def nms_jittor(boxes, scores, iou_threshold):
    """Jittor implementation of NMS."""
    # Simple NMS implementation
    keep = []
    order = scores.argsort(descending=True)
    
    while order.numel() > 0:
        if order.numel() == 1:
            keep.append(order[0])
            break
        
        i = order[0]
        keep.append(i)
        
        # Calculate IoU
        ious = box_iou_jittor(boxes[i:i+1], boxes[order[1:]])
        
        # Keep boxes with IoU less than threshold
        mask = ious[0] <= iou_threshold
        order = order[1:][mask]
    
    return jt.array(keep) if keep else jt.zeros(0, dtype=jt.int64)


def box_iou_jittor(box1, box2):
    """Calculate IoU between two sets of boxes."""
    # box1: [N, 4], box2: [M, 4]
    # Returns: [N, M]
    
    area1 = (box1[:, 2] - box1[:, 0]) * (box1[:, 3] - box1[:, 1])
    area2 = (box2[:, 2] - box2[:, 0]) * (box2[:, 3] - box2[:, 1])
    
    # Intersection
    lt = jt.maximum(box1[:, None, :2], box2[:, :2])  # [N, M, 2]
    rb = jt.minimum(box1[:, None, 2:], box2[:, 2:])  # [N, M, 2]
    
    wh = (rb - lt).clamp(min_v=0)  # [N, M, 2]
    inter = wh[:, :, 0] * wh[:, :, 1]  # [N, M]
    
    # Union
    union = area1[:, None] + area2 - inter
    
    return inter / union

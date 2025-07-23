#!/usr/bin/env python3
# -*- coding:utf-8 -*-
"""
完全对齐PyTorch版本的后处理逻辑
包括NMS、坐标转换、置信度过滤等
"""

import jittor as jt
import jittor.nn as nn
import numpy as np
import time


def box_iou(box1, box2):
    """
    计算两个边界框的IoU - 对齐PyTorch版本
    Args:
        box1: [N, 4] (x1, y1, x2, y2)
        box2: [M, 4] (x1, y1, x2, y2)
    Returns:
        iou: [N, M]
    """
    # 获取交集区域
    inter_x1 = jt.maximum(box1[:, None, 0], box2[:, 0])
    inter_y1 = jt.maximum(box1[:, None, 1], box2[:, 1])
    inter_x2 = jt.minimum(box1[:, None, 2], box2[:, 2])
    inter_y2 = jt.minimum(box1[:, None, 3], box2[:, 3])
    
    # 计算交集面积
    inter_area = jt.clamp(inter_x2 - inter_x1, min_v=0) * jt.clamp(inter_y2 - inter_y1, min_v=0)
    
    # 计算各自面积
    box1_area = (box1[:, 2] - box1[:, 0]) * (box1[:, 3] - box1[:, 1])
    box2_area = (box2[:, 2] - box2[:, 0]) * (box2[:, 3] - box2[:, 1])
    
    # 计算并集面积
    union_area = box1_area[:, None] + box2_area - inter_area
    
    # 计算IoU
    iou = inter_area / jt.clamp(union_area, min_v=1e-6)
    
    return iou


def xywh2xyxy(x):
    """
    转换边界框格式从(x_center, y_center, width, height)到(x1, y1, x2, y2)
    对齐PyTorch版本
    """
    y = x.clone() if isinstance(x, jt.Var) else jt.array(x)
    y[:, 0] = x[:, 0] - x[:, 2] / 2  # x1
    y[:, 1] = x[:, 1] - x[:, 3] / 2  # y1
    y[:, 2] = x[:, 0] + x[:, 2] / 2  # x2
    y[:, 3] = x[:, 1] + x[:, 3] / 2  # y2
    return y


def xyxy2xywh(x):
    """
    转换边界框格式从(x1, y1, x2, y2)到(x_center, y_center, width, height)
    对齐PyTorch版本
    """
    y = x.clone() if isinstance(x, jt.Var) else jt.array(x)
    y[:, 0] = (x[:, 0] + x[:, 2]) / 2  # x_center
    y[:, 1] = (x[:, 1] + x[:, 3]) / 2  # y_center
    y[:, 2] = x[:, 2] - x[:, 0]        # width
    y[:, 3] = x[:, 3] - x[:, 1]        # height
    return y


def clip_coords(boxes, img_shape):
    """
    将边界框坐标限制在图像范围内 - 对齐PyTorch版本
    Args:
        boxes: [N, 4] (x1, y1, x2, y2)
        img_shape: (height, width)
    """
    boxes[:, 0] = jt.clamp(boxes[:, 0], min_v=0, max_v=img_shape[1])  # x1
    boxes[:, 1] = jt.clamp(boxes[:, 1], min_v=0, max_v=img_shape[0])  # y1
    boxes[:, 2] = jt.clamp(boxes[:, 2], min_v=0, max_v=img_shape[1])  # x2
    boxes[:, 3] = jt.clamp(boxes[:, 3], min_v=0, max_v=img_shape[0])  # y2


def scale_coords(img1_shape, coords, img0_shape, ratio_pad=None):
    """
    将坐标从img1_shape缩放到img0_shape - 对齐PyTorch版本
    Args:
        img1_shape: 当前图像尺寸 (height, width)
        coords: 坐标 [N, 4]
        img0_shape: 目标图像尺寸 (height, width)
        ratio_pad: (ratio, pad)
    """
    if ratio_pad is None:  # 从img1_shape计算
        gain = min(img1_shape[0] / img0_shape[0], img1_shape[1] / img0_shape[1])  # gain = old / new
        pad = (img1_shape[1] - img0_shape[1] * gain) / 2, (img1_shape[0] - img0_shape[0] * gain) / 2  # wh padding
    else:
        gain = ratio_pad[0][0]
        pad = ratio_pad[1]

    coords[:, [0, 2]] -= pad[0]  # x padding
    coords[:, [1, 3]] -= pad[1]  # y padding
    coords[:, :4] /= gain
    clip_coords(coords, img0_shape)
    return coords


def nms_jittor(boxes, scores, iou_threshold):
    """
    Jittor版本的NMS实现 - 对齐PyTorch版本
    Args:
        boxes: [N, 4] (x1, y1, x2, y2)
        scores: [N]
        iou_threshold: IoU阈值
    Returns:
        keep: 保留的索引
    """
    if len(boxes) == 0:
        return jt.array([], dtype=jt.int64)
    
    # 按分数排序
    _, indices = jt.argsort(scores, descending=True)
    
    keep = []
    while len(indices) > 0:
        # 选择分数最高的框
        current = indices[0]
        keep.append(current.item())
        
        if len(indices) == 1:
            break
            
        # 计算IoU
        current_box = boxes[current:current+1]
        other_boxes = boxes[indices[1:]]
        ious = box_iou(current_box, other_boxes).squeeze(0)
        
        # 保留IoU小于阈值的框
        mask = ious <= iou_threshold
        indices = indices[1:][mask]
    
    return jt.array(keep, dtype=jt.int64)


def non_max_suppression(prediction, conf_thres=0.25, iou_thres=0.45, classes=None, agnostic=False, multi_label=False,
                        labels=(), max_det=300, nm=0):
    """
    完全对齐PyTorch版本的NMS实现
    Args:
        prediction: [batch_size, num_boxes, 85] (x, y, w, h, conf, cls0, cls1, ...)
        conf_thres: 置信度阈值
        iou_thres: IoU阈值
        classes: 过滤的类别
        agnostic: 是否类别无关NMS
        multi_label: 是否多标签
        labels: GT标签
        max_det: 最大检测数
        nm: mask数量
    Returns:
        output: List[Tensor] 每个图像的检测结果 [num_det, 6] (x1, y1, x2, y2, conf, cls)
    """
    if isinstance(prediction, (list, tuple)):  # YOLOv5 model in validation model, output = (inference_out, loss_out)
        prediction = prediction[0]  # select only inference output

    device = prediction.device if hasattr(prediction, 'device') else 'cuda'
    mps = 'mps' in str(device)  # Apple MPS
    if mps:  # MPS not fully supported yet, convert tensors to CPU before NMS
        prediction = prediction.cpu()

    bs = prediction.shape[0]  # batch size
    nc = prediction.shape[2] - nm - 5  # number of classes
    xc = prediction[..., 4] > conf_thres  # candidates

    # Checks
    assert 0 <= conf_thres <= 1, f'Invalid Confidence threshold {conf_thres}, valid values are between 0.0 and 1.0'
    assert 0 <= iou_thres <= 1, f'Invalid IoU {iou_thres}, valid values are between 0.0 and 1.0'

    # Settings
    max_wh = 7680  # (pixels) maximum box width and height
    max_nms = 30000  # maximum number of boxes into torchvision.ops.nms()
    time_limit = 0.5 + 0.05 * bs  # seconds to quit after
    redundant = True  # require redundant detections
    multi_label &= nc > 1  # multiple labels per box (adds 0.5ms/img)
    merge = False  # use merge-NMS

    t = time.time()
    output = [jt.zeros((0, 6 + nm))] * bs
    
    for xi, x in enumerate(prediction):  # image index, image inference
        # Apply constraints
        x = x[xc[xi]]  # confidence

        # Cat apriori labels if autolabelling
        if labels and len(labels[xi]):
            lb = labels[xi]
            v = jt.zeros((len(lb), nc + nm + 5))
            v[:, :4] = lb[:, 1:5]  # box
            v[:, 4] = 1.0  # conf
            v[range(len(lb)), lb[:, 0].long() + 5] = 1.0  # cls
            x = jt.concat((x, v), 0)

        # If none remain process next image
        if not x.shape[0]:
            continue

        # Compute conf
        x[:, 5:] *= x[:, 4:5]  # conf = obj_conf * cls_conf

        # Box (center x, center y, width, height) to (x1, y1, x2, y2)
        box = xywh2xyxy(x[:, :4])
        mask = x[:, 5:5 + nc]

        # Detections matrix nx6 (xyxy, conf, cls)
        if multi_label:
            i, j = (mask > conf_thres).nonzero(as_tuple=False).T
            x = jt.concat((box[i], x[i, j + 5, None], j[:, None].float()), 1)
        else:  # best class only
            conf, j = mask.max(1, keepdims=True)
            x = jt.concat((box, conf, j.float()), 1)[conf.view(-1) > conf_thres]

        # Filter by class
        if classes is not None:
            x = x[(x[:, 5:6] == jt.array(classes)).any(1)]

        # Apply finite constraint
        # x = x[torch.isfinite(x).all(1)]

        # Check shape
        n = x.shape[0]  # number of boxes
        if not n:  # no boxes
            continue
        elif n > max_nms:  # excess boxes
            x = x[x[:, 4].argsort(descending=True)[:max_nms]]  # sort by confidence

        # Batched NMS
        c = x[:, 5:6] * (0 if agnostic else max_wh)  # classes
        boxes, scores = x[:, :4] + c, x[:, 4]  # boxes (offset by class), scores
        i = nms_jittor(boxes, scores, iou_thres)  # NMS
        if i.shape[0] > max_det:  # limit detections
            i = i[:max_det]
        if merge and (1 < n < 3E3):  # Merge NMS (boxes merged using weighted mean)
            # update boxes as boxes(i,4) = weights(i,n) * boxes(n,4)
            iou = box_iou(boxes[i], boxes) > iou_thres  # iou matrix
            weights = iou * scores[None]  # box weights
            x[i, :4] = jt.matmul(weights, x[:, :4]).float() / weights.sum(1, keepdims=True)  # merged boxes
            if redundant:
                i = i[iou.sum(1) > 1]  # require redundancy

        output[xi] = x[i]
        if mps:
            output[xi] = output[xi].to(device)
        if (time.time() - t) > time_limit:
            LOGGER.warning(f'WARNING ⚠️ NMS time limit {time_limit:.3f}s exceeded')
            break  # time limit exceeded

    return output


def letterbox(im, new_shape=(640, 640), color=(114, 114, 114), auto=True, scaleFill=False, scaleup=True, stride=32):
    """
    对齐PyTorch版本的letterbox实现
    Resize and pad image while meeting stride-multiple constraints
    """
    shape = im.shape[:2]  # current shape [height, width]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    # Scale ratio (new / old)
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    if not scaleup:  # only scale down, do not scale up (for better val mAP)
        r = min(r, 1.0)

    # Compute padding
    ratio = r, r  # width, height ratios
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding
    if auto:  # minimum rectangle
        dw, dh = np.mod(dw, stride), np.mod(dh, stride)  # wh padding
    elif scaleFill:  # stretch
        dw, dh = 0.0, 0.0
        new_unpad = (new_shape[1], new_shape[0])
        ratio = new_shape[1] / shape[1], new_shape[0] / shape[0]  # width, height ratios

    dw /= 2  # divide padding into 2 sides
    dh /= 2

    if shape[::-1] != new_unpad:  # resize
        im = cv2.resize(im, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    im = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # add border
    return im, ratio, (dw, dh)


# 导入cv2用于letterbox
import cv2

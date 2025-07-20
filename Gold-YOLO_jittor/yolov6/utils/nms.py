#!/usr/bin/env python3
# -*- coding:utf-8 -*-
"""
NMS and Post-processing for Jittor - 完全对齐PyTorch版本
"""

import jittor as jt
import numpy as np
import time


def non_max_suppression(prediction, conf_thres=0.25, iou_thres=0.45, classes=None, agnostic=False, multi_label=False,
                       labels=(), max_det=300, nm=0):
    """NMS后处理 - 完全对齐PyTorch版本"""
    
    # 检查输入
    if isinstance(prediction, (list, tuple)):
        prediction = prediction[0]
    
    device = prediction.device if hasattr(prediction, 'device') else 'cpu'
    mps = 'mps' in str(device)  # Apple MPS
    if mps:  # MPS不支持某些操作
        prediction = prediction.cpu()
    
    bs = prediction.shape[0]  # batch size
    nc = prediction.shape[2] - nm - 5  # number of classes
    xc = prediction[..., 4] > conf_thres  # candidates
    
    # 设置
    max_wh = 7680  # (pixels) maximum box width and height
    max_nms = 30000  # maximum number of boxes into torchvision.ops.nms()
    redundant = True  # require redundant detections
    multi_label &= nc > 1  # multiple labels per box (adds 0.5ms/img)
    merge = False  # use merge-NMS
    
    t = time.time()
    mi = 5 + nc  # mask start index
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
        
        # Box/Mask
        box = xywh2xyxy(x[:, :4])  # center_x, center_y, width, height) to (x1, y1, x2, y2)
        mask = x[:, mi:]  # zero columns if no masks
        
        # Detections matrix nx6 (xyxy, conf, cls)
        if multi_label:
            i, j = (x[:, 5:mi] > conf_thres).nonzero(as_tuple=False).T
            x = jt.concat((box[i], x[i, 5 + j, None], j[:, None].float(), mask[i]), 1)
        else:  # best class only
            # 完全对齐PyTorch版本，处理Jittor的返回值差异
            cls_scores = x[:, 5:mi]  # [n, num_classes]

            # 处理max返回值
            max_result = cls_scores.max(1)
            if isinstance(max_result, tuple):
                conf = max_result[0].unsqueeze(1)  # [n, 1]
            else:
                conf = max_result.unsqueeze(1)

            # 处理argmax返回值
            argmax_result = cls_scores.argmax(1)
            if isinstance(argmax_result, tuple):
                j = argmax_result[0].unsqueeze(1).float()  # [n, 1]
            else:
                j = argmax_result.unsqueeze(1).float()

            x = jt.concat((box, conf, j, mask), 1)[conf.view(-1) > conf_thres]
        
        # Filter by class
        if classes is not None:
            x = x[(x[:, 5:6] == jt.array(classes)).any(1)]
        
        # Apply finite constraint
        # if not torch.isfinite(x).all():
        #     x = x[torch.isfinite(x).all(1)]
        
        # Check shape
        n = x.shape[0]  # number of boxes
        if not n:  # no boxes
            continue
        x = x[x[:, 4].argsort(descending=True)[:max_nms]]  # sort by confidence and remove excess boxes
        
        # Batched NMS
        # 检查x的形状并处理
        if x.numel() == 0 or x.shape[0] == 0:
            continue

        if len(x.shape) < 2 or x.shape[1] < 6:
            print(f"Warning: x shape {x.shape} invalid, skipping NMS")
            continue

        c = x[:, 5:6] * (0 if agnostic else max_wh)  # classes
        boxes, scores = x[:, :4] + c, x[:, 4]  # boxes (offset by class), scores
        i = nms_jittor(boxes, scores, iou_thres)  # NMS
        i = i[:max_det]  # limit detections
        if merge and (1 < n < 3E3):  # Merge NMS (boxes merged using weighted mean)
            # update boxes as boxes(i,4) = weights(i,n) * boxes(n,4)
            iou = box_iou(boxes[i], boxes) > iou_thres  # iou matrix
            weights = iou * scores[None]  # box weights
            x[i, :4] = jt.mm(weights, x[:, :4]).float() / weights.sum(1, keepdim=True)  # merged boxes
            if redundant:
                i = i[iou.sum(1) > 1]  # require redundancy
        
        output[xi] = x[i]
        if mps:
            output[xi] = output[xi].to(device)
        if (time.time() - t) > 0.5 + 0.05 * bs:
            print(f'WARNING ⚠️ NMS time limit {0.5 + 0.05 * bs:.3f}s exceeded')
            break  # time limit exceeded
    
    return output


def nms_jittor(boxes, scores, iou_threshold):
    """Jittor实现的NMS"""
    if boxes.numel() == 0:
        return jt.empty((0,), dtype=jt.int64)
    
    # 按分数排序
    sorted_indices = scores.argsort(descending=True)
    
    keep = []
    while len(sorted_indices) > 0:
        # 选择分数最高的框
        current = sorted_indices[0]
        keep.append(current)
        
        if len(sorted_indices) == 1:
            break
        
        # 计算IoU
        current_box = boxes[current].unsqueeze(0)
        other_boxes = boxes[sorted_indices[1:]]
        
        ious = box_iou(current_box, other_boxes).squeeze(0)
        
        # 保留IoU小于阈值的框
        mask = ious <= iou_threshold
        sorted_indices = sorted_indices[1:][mask]
    
    return jt.array(keep, dtype=jt.int64)


def xywh2xyxy(x):
    """Convert nx4 boxes from [x, y, w, h] to [x1, y1, x2, y2]"""
    y = x.clone() if isinstance(x, jt.Var) else jt.array(x)
    y[..., 0] = x[..., 0] - x[..., 2] / 2  # top left x
    y[..., 1] = x[..., 1] - x[..., 3] / 2  # top left y
    y[..., 2] = x[..., 0] + x[..., 2] / 2  # bottom right x
    y[..., 3] = x[..., 1] + x[..., 3] / 2  # bottom right y
    return y


def box_iou(box1, box2, eps=1e-7):
    """计算IoU"""
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


# 向后兼容
def non_max_suppression_jittor(prediction, conf_thres=0.25, iou_thres=0.45, classes=None, agnostic=False, multi_label=False,
                              labels=(), max_det=300):
    """向后兼容的NMS函数"""
    return non_max_suppression(prediction, conf_thres, iou_thres, classes, agnostic, multi_label, labels, max_det)

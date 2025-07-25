#!/usr/bin/env python3
# -*- coding:utf-8 -*-
"""
GOLD-YOLO Jittor版本 - NMS非极大值抑制模块
从PyTorch版本迁移到Jittor框架，严格对齐所有功能
"""

import os
import time
import numpy as np
import cv2
import jittor as jt


# Settings
np.set_printoptions(linewidth=320, formatter={'float_kind': '{:11.5g}'.format})  # format short g, %precision=5
cv2.setNumThreads(0)  # prevent OpenCV from multithreading (incompatible with DataLoader)
os.environ['NUMEXPR_MAX_THREADS'] = str(min(os.cpu_count(), 8))  # NumExpr max threads


def xywh2xyxy(x):
    '''Convert boxes with shape [n, 4] from [x, y, w, h] to [x1, y1, x2, y2] where x1y1 is top-left, x2y2=bottom-right.'''
    y = x.clone() if isinstance(x, jt.Var) else np.copy(x)
    y[:, 0] = x[:, 0] - x[:, 2] / 2  # top left x
    y[:, 1] = x[:, 1] - x[:, 3] / 2  # top left y
    y[:, 2] = x[:, 0] + x[:, 2] / 2  # bottom right x
    y[:, 3] = x[:, 1] + x[:, 3] / 2  # bottom right y
    return y


def xyxy2xywh(x):
    '''Convert boxes with shape [n, 4] from [x1, y1, x2, y2] to [x, y, w, h] where x1y1 is top-left, x2y2=bottom-right.'''
    y = x.clone() if isinstance(x, jt.Var) else np.copy(x)
    y[:, 0] = (x[:, 0] + x[:, 2]) / 2  # x center
    y[:, 1] = (x[:, 1] + x[:, 3]) / 2  # y center
    y[:, 2] = x[:, 2] - x[:, 0]  # width
    y[:, 3] = x[:, 3] - x[:, 1]  # height
    return y


def box_iou(box1, box2):
    """
    Return intersection-over-union (Jaccard index) of boxes.
    Both sets of boxes are expected to be in (x1, y1, x2, y2) format.
    Arguments:
        box1 (Tensor[N, 4])
        box2 (Tensor[M, 4])
    Returns:
        iou (Tensor[N, M]): the NxM matrix containing the pairwise IoU values for every element in boxes1 and boxes2
    """
    
    def box_area(box):
        # box = 4xn
        return (box[2] - box[0]) * (box[3] - box[1])
    
    area1 = box_area(box1.transpose(1, 0))  # Jittor使用transpose替代.T
    area2 = box_area(box2.transpose(1, 0))
    
    # inter(N,M) = (rb(N,M,2) - lt(N,M,2)).clamp(0).prod(2)
    inter = (jt.minimum(box1[:, None, 2:], box2[:, 2:]) - jt.maximum(box1[:, None, :2], box2[:, :2])).clamp(0).prod(2)
    return inter / (area1[:, None] + area2 - inter)  # iou = inter / (area1 + area2 - inter)


def jittor_nms(boxes, scores, iou_threshold):
    """
    Jittor实现的NMS算法
    
    Args:
        boxes: [N, 4] 边界框坐标 (x1, y1, x2, y2)
        scores: [N] 置信度分数
        iou_threshold: IoU阈值
    
    Returns:
        keep: 保留的索引
    """
    if boxes.numel() == 0:
        return jt.empty((0,), dtype=jt.int64)
    
    # 按分数排序
    _, order = scores.sort(descending=True)
    
    keep = []
    while order.numel() > 0:
        # 保留分数最高的框
        i = order[0]
        keep.append(i)
        
        if order.numel() == 1:
            break
        
        # 计算IoU
        iou = box_iou(boxes[i:i+1], boxes[order[1:]])[0]
        
        # 保留IoU小于阈值的框
        mask = iou <= iou_threshold
        order = order[1:][mask]
    
    return jt.array(keep).long()  # Jittor的array使用方式


def non_max_suppression(prediction, conf_thres=0.25, iou_thres=0.45, classes=None, agnostic=False, multi_label=False, max_det=300):
    """Runs Non-Maximum Suppression (NMS) on inference results.
    This code is borrowed from: https://github.com/ultralytics/yolov5/blob/47233e1698b89fc437a4fb9463c815e9171be955/utils/general.py#L775
    Args:
        prediction: (tensor), with shape [N, 5 + num_classes], N is the number of bboxes.
        conf_thres: (float) confidence threshold.
        iou_thres: (float) iou threshold.
        classes: (None or list[int]), if a list is provided, nms only keep the classes you provide.
        agnostic: (bool), when it is set to True, we do class-independent nms, otherwise, different class would do nms respectively.
        multi_label: (bool), when it is set to True, one box can have multi labels, otherwise, one box only huave one label.
        max_det:(int), max number of output bboxes.

    Returns:
         list of detections, echo item is one tensor with shape (num_boxes, 6), 6 is for [xyxy, conf, cls].
    """

    num_classes = prediction.shape[2] - 5  # number of classes
    max_cls_scores = prediction[..., 5:].max(dim=-1)
    if isinstance(max_cls_scores, tuple):
        max_cls_scores = max_cls_scores[0]
    pred_candidates = jt.logical_and(prediction[..., 4] > conf_thres, max_cls_scores > conf_thres)  # candidates
    # Check the parameters.
    assert 0 <= conf_thres <= 1, f'conf_thresh must be in 0.0 to 1.0, however {conf_thres} is provided.'
    assert 0 <= iou_thres <= 1, f'iou_thres must be in 0.0 to 1.0, however {iou_thres} is provided.'

    # Function settings.
    max_wh = 4096  # maximum box width and height
    max_nms = 30000  # maximum number of boxes put into nms()
    time_limit = 10.0  # quit the function when nms cost time exceed the limit time.
    multi_label &= num_classes > 1  # multiple labels per box

    tik = time.time()
    output = [jt.zeros((0, 6))] * prediction.shape[0]
    for img_idx, x in enumerate(prediction):  # image index, image inference
        x = x[pred_candidates[img_idx]]  # confidence

        # If no box remains, skip the next process.
        if not x.shape[0]:
            continue

        # confidence multiply the objectness
        x[:, 5:] *= x[:, 4:5]  # conf = obj_conf * cls_conf

        # (center x, center y, width, height) to (x1, y1, x2, y2)
        box = xywh2xyxy(x[:, :4])

        # Detections matrix's shape is  (n,6), each row represents (xyxy, conf, cls)
        if multi_label:
            box_idx, class_idx = (x[:, 5:] > conf_thres).nonzero().T
            x = jt.concat((box[box_idx], x[box_idx, class_idx + 5, None], class_idx[:, None].float()), 1)
        else:  # Only keep the class with highest scores.
            max_result = x[:, 5:].max(1, keepdim=True)
            if isinstance(max_result, tuple):
                conf, class_idx = max_result
            else:
                conf = max_result
                argmax_result = x[:, 5:].argmax(1)
                if isinstance(argmax_result, tuple):
                    class_idx = argmax_result[1].unsqueeze(1)  # 取索引部分
                else:
                    class_idx = argmax_result.unsqueeze(1)
            x = jt.concat((box, conf, class_idx.float()), 1)[conf.view(-1) > conf_thres]

        # Filter by class, only keep boxes whose category is in classes.
        if classes is not None:
            x = x[(x[:, 5:6] == jt.Var(classes).unsqueeze(0)).any(1)]  # 使用jt.Var替代jt.array

        # Check shape
        num_box = x.shape[0]  # number of boxes
        if not num_box:  # no boxes kept.
            continue
        elif num_box > max_nms:  # excess max boxes' number.
            x = x[x[:, 4].argsort(descending=True)[:max_nms]]  # sort by confidence

        # Batched NMS
        class_offset = x[:, 5:6] * (0 if agnostic else max_wh)  # classes
        boxes, scores = x[:, :4] + class_offset, x[:, 4]  # boxes (offset by class), scores
        keep_box_idx = jittor_nms(boxes, scores, iou_thres)  # NMS
        if keep_box_idx.shape[0] > max_det:  # limit detections
            keep_box_idx = keep_box_idx[:max_det]

        output[img_idx] = x[keep_box_idx]
        if (time.time() - tik) > time_limit:
            print(f'WARNING: NMS time limit {time_limit}s exceeded')
            break  # time limit exceeded

    return output


def clip_coords(boxes, img_shape):
    """Clip bounding xyxy bounding boxes to image shape (height, width)"""
    boxes[:, 0].clamp_(0, img_shape[1])  # x1
    boxes[:, 1].clamp_(0, img_shape[0])  # y1
    boxes[:, 2].clamp_(0, img_shape[1])  # x2
    boxes[:, 3].clamp_(0, img_shape[0])  # y2


def scale_coords(img1_shape, coords, img0_shape, ratio_pad=None):
    """Rescale coords (xyxy) from img1_shape to img0_shape"""
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


def soft_nms(boxes, scores, sigma=0.5, thresh=0.001, cuda=0):
    """
    Soft NMS implementation
    
    Args:
        boxes: [N, 4] 边界框
        scores: [N] 分数
        sigma: 高斯函数的标准差
        thresh: 分数阈值
        cuda: 是否使用CUDA（Jittor自动处理）
    
    Returns:
        keep: 保留的索引
    """
    N = boxes.shape[0]
    indexes = jt.arange(0, N, dtype=jt.float32)
    
    for i in range(N):
        # 找到当前最高分数的框
        max_score_index = scores[i:].argmax()
        max_score_index += i
        
        # 交换到当前位置
        boxes[[i, max_score_index]] = boxes[[max_score_index, i]]
        scores[[i, max_score_index]] = scores[[max_score_index, i]]
        indexes[[i, max_score_index]] = indexes[[max_score_index, i]]
        
        # 计算IoU
        if i < N - 1:
            iou = box_iou(boxes[i:i+1], boxes[i+1:])[0]
            
            # 应用软NMS
            weight = jt.exp(-(iou * iou) / sigma)
            scores[i+1:] *= weight
    
    # 过滤低分数的框
    keep = scores > thresh
    return indexes[keep].long()


def batched_nms(boxes, scores, idxs, iou_threshold):
    """
    批量NMS，对不同类别分别进行NMS
    
    Args:
        boxes: [N, 4] 边界框
        scores: [N] 分数
        idxs: [N] 类别索引
        iou_threshold: IoU阈值
    
    Returns:
        keep: 保留的索引
    """
    if boxes.numel() == 0:
        return jt.empty((0,), dtype=jt.int64)
    
    # 为不同类别添加偏移，避免不同类别间的NMS
    max_coordinate = boxes.max()
    offsets = idxs.astype(boxes.dtype) * (max_coordinate + 1)
    boxes_for_nms = boxes + offsets[:, None]
    
    keep = jittor_nms(boxes_for_nms, scores, iou_threshold)
    return keep

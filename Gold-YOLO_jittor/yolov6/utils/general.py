#!/usr/bin/env python3
# -*- coding:utf-8 -*-
"""
GOLD-YOLO Jittor版本 - 通用工具函数
从PyTorch版本迁移到Jittor框架
"""

import os
import glob
import math
import jittor as jt
from pathlib import Path


def check_img_size(img_size, s=32, floor=0):
    """Verify image size is a multiple of stride s in each dimension"""
    if isinstance(img_size, int):  # integer i.e. img_size=640
        new_size = max(make_divisible(img_size, int(s)), floor)
    else:  # list i.e. img_size=[640, 480]
        new_size = [max(make_divisible(x, int(s)), floor) for x in img_size]
    if new_size != img_size:
        print(f'WARNING: --img-size {img_size} must be multiple of max stride {s}, updating to {new_size}')
    return new_size

def make_divisible(x, divisor):
    """Returns nearest x divisible by divisor"""
    if isinstance(divisor, jt.Var):
        divisor = int(divisor.max())  # to int
    return math.ceil(x / divisor) * divisor

def increment_name(path):
    '''increase save directory's id'''
    path = Path(path)
    sep = ''
    if path.exists():
        path, suffix = (path.with_suffix(''), path.suffix) if path.is_file() else (path, '')
        for n in range(1, 9999):
            p = f'{path}{sep}{n}{suffix}'
            if not os.path.exists(p):
                break
        path = Path(p)
    return path


def find_latest_checkpoint(search_dir='.'):
    '''Find the most recent saved checkpoint in search_dir.'''
    checkpoint_list = glob.glob(f'{search_dir}/**/last*.pkl', recursive=True)  # Jittor使用.pkl格式
    return max(checkpoint_list, key=os.path.getctime) if checkpoint_list else ''


def dist2bbox(distance, anchor_points, box_format='xyxy'):
    '''Transform distance(ltrb) to box(xywh or xyxy).'''
    lt, rb = jt.split(distance, 2, -1)  # 使用jt.split替代torch.split
    x1y1 = anchor_points - lt
    x2y2 = anchor_points + rb
    if box_format == 'xyxy':
        bbox = jt.concat([x1y1, x2y2], -1)  # 使用jt.concat替代torch.cat
    elif box_format == 'xywh':
        c_xy = (x1y1 + x2y2) / 2
        wh = x2y2 - x1y1
        bbox = jt.concat([c_xy, wh], -1)
    return bbox


def bbox2dist(anchor_points, bbox, reg_max):
    '''Transform bbox(xyxy) to dist(ltrb).'''
    x1y1, x2y2 = jt.split(bbox, 2, -1)
    lt = anchor_points - x1y1
    rb = x2y2 - anchor_points
    dist = jt.concat([lt, rb], -1).clamp(0, reg_max - 0.01)  # 使用clamp替代clip
    return dist


def xywh2xyxy(bboxes):
    '''Transform bbox(xywh) to box(xyxy).'''
    # 修复坐标转换bug - 保存原始中心坐标和尺寸
    cx, cy, w, h = bboxes[..., 0], bboxes[..., 1], bboxes[..., 2], bboxes[..., 3]
    bboxes[..., 0] = cx - w * 0.5  # x1 = cx - w/2
    bboxes[..., 1] = cy - h * 0.5  # y1 = cy - h/2
    bboxes[..., 2] = cx + w * 0.5  # x2 = cx + w/2
    bboxes[..., 3] = cy + h * 0.5  # y2 = cy + h/2
    return bboxes


def box_iou(box1, box2):
    # https://github.com/pytorch/vision/blob/master/torchvision/ops/boxes.py
    """
    Return intersection-over-union (Jaccard index) of boxes.
    Both sets of boxes are expected to be in (x1, y1, x2, y2) format.
    Arguments:
        box1 (Tensor[N, 4])
        box2 (Tensor[M, 4])
    Returns:
        iou (Tensor[N, M]): the NxM matrix containing the pairwise
            IoU values for every element in boxes1 and boxes2
    """

    def box_area(box):
        # box = 4xn
        return (box[2] - box[0]) * (box[3] - box[1])

    area1 = box_area(box1.T)
    area2 = box_area(box2.T)

    # inter(N,M) = (rb(N,M,2) - lt(N,M,2)).clamp(0).prod(2)
    inter = (jt.minimum(box1[:, None, 2:], box2[:, 2:]) - jt.maximum(box1[:, None, :2], box2[:, :2])).clamp(0).prod(2)
    return inter / (area1[:, None] + area2 - inter)  # iou = inter / (area1 + area2 - inter)


def xyxy2xywh(x):
    """Convert nx4 boxes from [x1, y1, x2, y2] to [x, y, w, h] where xy1=top-left, xy2=bottom-right"""
    y = x.clone() if isinstance(x, jt.Var) else jt.array(x)
    y[:, 0] = (x[:, 0] + x[:, 2]) / 2  # x center
    y[:, 1] = (x[:, 1] + x[:, 3]) / 2  # y center
    y[:, 2] = x[:, 2] - x[:, 0]  # width
    y[:, 3] = x[:, 3] - x[:, 1]  # height
    return y


def xywhn2xyxy(x, w=640, h=640, padw=0, padh=0):
    """Convert nx4 boxes from [x, y, w, h] normalized to [x1, y1, x2, y2] where xy1=top-left, xy2=bottom-right"""
    y = x.clone() if isinstance(x, jt.Var) else jt.array(x)
    y[:, 0] = w * (x[:, 0] - x[:, 2] / 2) + padw  # top left x
    y[:, 1] = h * (x[:, 1] - x[:, 3] / 2) + padh  # top left y
    y[:, 2] = w * (x[:, 0] + x[:, 2] / 2) + padw  # bottom right x
    y[:, 3] = h * (x[:, 1] + x[:, 3] / 2) + padh  # bottom right y
    return y


def xyxy2xywhn(x, w=640, h=640, clip=False, eps=0.0):
    """Convert nx4 boxes from [x1, y1, x2, y2] to [x, y, w, h] normalized where xy1=top-left, xy2=bottom-right"""
    if clip:
        clip_coords(x, (h - eps, w - eps))  # warning: inplace clip
    y = x.clone() if isinstance(x, jt.Var) else jt.array(x)
    y[:, 0] = ((x[:, 0] + x[:, 2]) / 2) / w  # x center
    y[:, 1] = ((x[:, 1] + x[:, 3]) / 2) / h  # y center
    y[:, 2] = (x[:, 2] - x[:, 0]) / w  # width
    y[:, 3] = (x[:, 3] - x[:, 1]) / h  # height
    return y


def clip_coords(boxes, shape):
    """Clip bounding xyxy bounding boxes to image shape (height, width)"""
    if isinstance(boxes, jt.Var):
        boxes[:, 0].clamp_(0, shape[1])  # x1
        boxes[:, 1].clamp_(0, shape[0])  # y1
        boxes[:, 2].clamp_(0, shape[1])  # x2
        boxes[:, 3].clamp_(0, shape[0])  # y2
    else:  # np.array (faster grouped)
        boxes[:, [0, 2]] = boxes[:, [0, 2]].clip(0, shape[1])  # x1, x2
        boxes[:, [1, 3]] = boxes[:, [1, 3]].clip(0, shape[0])  # y1, y2


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

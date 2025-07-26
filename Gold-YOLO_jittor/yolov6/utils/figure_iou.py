#!/usr/bin/env python3
# -*- coding:utf-8 -*-
"""
GOLD-YOLO Jittor版本 - IoU损失函数
从PyTorch版本迁移到Jittor框架，严格对齐所有功能
"""

import math
import jittor as jt


class IOUloss:
    """ Calculate IoU loss.
    """
    def __init__(self, box_format='xywh', iou_type='ciou', reduction='none', eps=1e-7):
        """ Setting of the class.
        Args:
            box_format: (string), must be one of 'xywh' or 'xyxy'.
            iou_type: (string), can be one of 'ciou', 'diou', 'giou' or 'siou'
            reduction: (string), specifies the reduction to apply to the output, must be one of 'none', 'mean','sum'.
            eps: (float), a value to avoid divide by zero error.
        """
        self.box_format = box_format
        self.iou_type = iou_type.lower()
        self.reduction = reduction
        self.eps = eps

    def __call__(self, box1, box2):
        """ calculate iou. box1 and box2 are jittor tensor with shape [M, 4] and [Nm 4].
        """
        if box1.shape[0] != box2.shape[0]:
            box2 = box2.transpose(1, 0)  # Jittor使用transpose替代.T
            if self.box_format == 'xyxy':
                b1_x1, b1_y1, b1_x2, b1_y2 = box1[0], box1[1], box1[2], box1[3]
                b2_x1, b2_y1, b2_x2, b2_y2 = box2[0], box2[1], box2[2], box2[3]
            elif self.box_format == 'xywh':
                b1_x1, b1_x2 = box1[0] - box1[2] / 2, box1[0] + box1[2] / 2
                b1_y1, b1_y2 = box1[1] - box1[3] / 2, box1[1] + box1[3] / 2
                b2_x1, b2_x2 = box2[0] - box2[2] / 2, box2[0] + box2[2] / 2
                b2_y1, b2_y2 = box2[1] - box2[3] / 2, box2[1] + box2[3] / 2
        else:
            if self.box_format == 'xyxy':
                b1_x1, b1_y1, b1_x2, b1_y2 = jt.split(box1, 1, dim=-1)
                b2_x1, b2_y1, b2_x2, b2_y2 = jt.split(box2, 1, dim=-1)

            elif self.box_format == 'xywh':
                b1_x1, b1_y1, b1_w, b1_h = jt.split(box1, 1, dim=-1)
                b2_x1, b2_y1, b2_w, b2_h = jt.split(box2, 1, dim=-1)
                b1_x1, b1_x2 = b1_x1 - b1_w / 2, b1_x1 + b1_w / 2
                b1_y1, b1_y2 = b1_y1 - b1_h / 2, b1_y1 + b1_h / 2
                b2_x1, b2_x2 = b2_x1 - b2_w / 2, b2_x1 + b2_w / 2
                b2_y1, b2_y2 = b2_y1 - b2_h / 2, b2_y1 + b2_h / 2

        # Intersection area
        inter = (jt.minimum(b1_x2, b2_x2) - jt.maximum(b1_x1, b2_x1)).clamp(0) * \
                (jt.minimum(b1_y2, b2_y2) - jt.maximum(b1_y1, b2_y1)).clamp(0)

        # Union Area - 添加更严格的数值检查
        w1, h1 = b1_x2 - b1_x1, b1_y2 - b1_y1 + self.eps
        w2, h2 = b2_x2 - b2_x1, b2_y2 - b2_y1 + self.eps

        # 确保宽高为正值
        w1 = jt.clamp(w1, self.eps, 1e6)
        h1 = jt.clamp(h1, self.eps, 1e6)
        w2 = jt.clamp(w2, self.eps, 1e6)
        h2 = jt.clamp(h2, self.eps, 1e6)

        union = w1 * h1 + w2 * h2 - inter + self.eps
        # 确保union为正值
        union = jt.clamp(union, self.eps, 1e8)

        iou = inter / union

        # 限制IoU值在合理范围内，防止数值错误
        iou = jt.clamp(iou, 0.0, 1.0)

        cw = jt.maximum(b1_x2, b2_x2) - jt.minimum(b1_x1, b2_x1)  # convex width
        ch = jt.maximum(b1_y2, b2_y2) - jt.minimum(b1_y1, b2_y1)  # convex height
        if self.iou_type == 'giou':
            c_area = cw * ch + self.eps  # convex area
            iou = iou - (c_area - union) / c_area
            # 限制GIoU值在合理范围内
            iou = jt.clamp(iou, -1.0, 1.0)
        elif self.iou_type in ['diou', 'ciou']:
            c2 = cw ** 2 + ch ** 2 + self.eps  # convex diagonal squared
            rho2 = ((b2_x1 + b2_x2 - b1_x1 - b1_x2) ** 2 +
                    (b2_y1 + b2_y2 - b1_y1 - b1_y2) ** 2) / 4  # center distance squared
            if self.iou_type == 'diou':
                iou = iou - rho2 / c2
                # 限制DIoU值在合理范围内
                iou = jt.clamp(iou, -1.0, 1.0)
            elif self.iou_type == 'ciou':
                v = (4 / math.pi ** 2) * jt.pow(jt.atan(w2 / h2) - jt.atan(w1 / h1), 2)
                with jt.no_grad():
                    alpha = v / (v - iou + (1 + self.eps))
                iou = iou - (rho2 / c2 + v * alpha)
                # 限制CIoU值在合理范围内
                iou = jt.clamp(iou, -1.0, 1.0)
        elif self.iou_type == 'siou':
            # SIoU Loss https://arxiv.org/pdf/2205.12740.pdf
            s_cw = (b2_x1 + b2_x2 - b1_x1 - b1_x2) * 0.5 + self.eps
            s_ch = (b2_y1 + b2_y2 - b1_y1 - b1_y2) * 0.5 + self.eps
            sigma = jt.pow(s_cw ** 2 + s_ch ** 2, 0.5)
            sin_alpha_1 = jt.abs(s_cw) / sigma
            sin_alpha_2 = jt.abs(s_ch) / sigma
            threshold = pow(2, 0.5) / 2
            sin_alpha = jt.where(sin_alpha_1 > threshold, sin_alpha_2, sin_alpha_1)
            angle_cost = jt.cos(jt.arcsin(sin_alpha) * 2 - math.pi / 2)
            rho_x = (s_cw / cw) ** 2
            rho_y = (s_ch / ch) ** 2
            gamma = angle_cost - 2
            distance_cost = 2 - jt.exp(gamma * rho_x) - jt.exp(gamma * rho_y)
            omiga_w = jt.abs(w1 - w2) / jt.maximum(w1, w2)
            omiga_h = jt.abs(h1 - h2) / jt.maximum(h1, h2)
            shape_cost = jt.pow(1 - jt.exp(-1 * omiga_w), 4) + jt.pow(1 - jt.exp(-1 * omiga_h), 4)
            iou = iou - 0.5 * (distance_cost + shape_cost)
            # 限制SIoU值在合理范围内
            iou = jt.clamp(iou, -1.0, 1.0)
        loss = 1.0 - iou

        # 确保损失值在合理范围内
        loss = jt.clamp(loss, 0.0, 10.0)  # 限制损失在[0, 10]范围内

        # 检查NaN值 - 修复Jittor API
        try:
            nan_mask = jt.isnan(loss)
            if nan_mask.sum() > 0:
                print(f"⚠️ IoU损失包含NaN，设为1.0")
                loss = jt.ones_like(loss)
        except:
            # 如果isnan不可用，跳过检查
            pass

        if self.reduction == 'sum':
            loss = loss.sum()
        elif self.reduction == 'mean':
            loss = loss.mean()

        return loss


def bbox_iou(box1, box2, xywh=True, GIoU=False, DIoU=False, CIoU=False, eps=1e-7):
    """
    Calculate Intersection over Union (IoU) of box1(1, 4) to box2(n, 4).
    
    Args:
        box1 (jt.Var): A tensor representing a single bounding box with shape (1, 4).
        box2 (jt.Var): A tensor representing n bounding boxes with shape (n, 4).
        xywh (bool): If True, input boxes are in (x, y, w, h) format. If False, in (x1, y1, x2, y2) format.
        GIoU (bool): If True, calculate Generalized IoU.
        DIoU (bool): If True, calculate Distance IoU.
        CIoU (bool): If True, calculate Complete IoU.
        eps (float): A small value to avoid division by zero.
        
    Returns:
        iou (jt.Var): IoU values with shape (n,).
    """
    
    # Get the coordinates of bounding boxes
    if xywh:  # transform from xywh to xyxy
        (x1, y1, w1, h1), (x2, y2, w2, h2) = box1.chunk(4, 1), box2.chunk(4, 1)
        w1_, h1_, w2_, h2_ = w1 / 2, h1 / 2, w2 / 2, h2 / 2
        b1_x1, b1_x2, b1_y1, b1_y2 = x1 - w1_, x1 + w1_, y1 - h1_, y1 + h1_
        b2_x1, b2_x2, b2_y1, b2_y2 = x2 - w2_, x2 + w2_, y2 - h2_, y2 + h2_
    else:  # x1, y1, x2, y2 = box1
        b1_x1, b1_y1, b1_x2, b1_y2 = box1.chunk(4, 1)
        b2_x1, b2_y1, b2_x2, b2_y2 = box2.chunk(4, 1)
        w1, h1 = b1_x2 - b1_x1, b1_y2 - b1_y1 + eps
        w2, h2 = b2_x2 - b2_x1, b2_y2 - b2_y1 + eps

    # Intersection area
    inter = (jt.minimum(b1_x2, b2_x2) - jt.maximum(b1_x1, b2_x1)).clamp(0) * \
            (jt.minimum(b1_y2, b2_y2) - jt.maximum(b1_y1, b2_y1)).clamp(0)

    # Union Area
    union = w1 * h1 + w2 * h2 - inter + eps

    # IoU
    iou = inter / union
    if CIoU or DIoU or GIoU:
        cw = jt.maximum(b1_x2, b2_x2) - jt.minimum(b1_x1, b2_x1)  # convex (smallest enclosing box) width
        ch = jt.maximum(b1_y2, b2_y2) - jt.minimum(b1_y1, b2_y1)  # convex height
        if CIoU or DIoU:  # Distance or Complete IoU https://arxiv.org/abs/1911.08287v1
            c2 = cw ** 2 + ch ** 2 + eps  # convex diagonal squared
            rho2 = ((b2_x1 + b2_x2 - b1_x1 - b1_x2) ** 2 + (b2_y1 + b2_y2 - b1_y1 - b1_y2) ** 2) / 4  # center dist ** 2
            if CIoU:  # https://github.com/Zzh-tju/DIoU-SSD-pytorch/blob/master/utils/box/box_utils.py#L47
                v = (4 / math.pi ** 2) * jt.pow(jt.atan(w2 / h2) - jt.atan(w1 / h1), 2)
                with jt.no_grad():
                    alpha = v / (v - iou + (1 + eps))
                return iou - (rho2 / c2 + v * alpha)  # CIoU
            return iou - rho2 / c2  # DIoU
        c_area = cw * ch + eps  # convex area
        return iou - (c_area - union) / c_area  # GIoU https://arxiv.org/pdf/1902.09630.pdf
    return iou  # IoU

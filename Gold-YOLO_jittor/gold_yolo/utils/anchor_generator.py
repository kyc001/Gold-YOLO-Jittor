#!/usr/bin/env python3
# -*- coding:utf-8 -*-
"""
Anchor生成器 - 简化版
"""

import jittor as jt

def generate_anchors(feats, fpn_strides, grid_cell_size=5.0, grid_cell_offset=0.5, device='cuda'):
    """生成anchors - 简化实现"""
    anchors = []
    anchor_points = []
    num_anchors_list = []
    stride_tensor = []
    
    for i, (feat, stride) in enumerate(zip(feats, fpn_strides)):
        _, _, h, w = feat.shape
        shift_x = jt.arange(0, w, dtype=jt.float32) + grid_cell_offset
        shift_y = jt.arange(0, h, dtype=jt.float32) + grid_cell_offset
        shift_y, shift_x = jt.meshgrid(shift_y, shift_x)
        
        anchor_point = jt.stack([shift_x, shift_y], dim=-1).reshape(-1, 2)
        anchor_points.append(anchor_point * stride)
        num_anchors_list.append(len(anchor_point))
        stride_tensor.append(jt.full((len(anchor_point), 1), stride, dtype=jt.float32))
    
    anchor_points = jt.concat(anchor_points, dim=0)
    stride_tensor = jt.concat(stride_tensor, dim=0)
    
    return anchor_points, stride_tensor
